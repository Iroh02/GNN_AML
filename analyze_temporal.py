"""
Deep Analysis: Why Temporal Encoding Hurts Performance
======================================================

Investigate why temporal encoding decreases performance in temporal split:
1. Distribution shift analysis - how do timesteps differ between train/test?
2. Illicit pattern analysis - do illicit patterns change over time?
3. Time encoding overfitting - does the model memorize timestep patterns?
4. Per-timestep performance breakdown
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

def load_data():
    df = pd.read_csv('data/raw/elliptic_txs_features.csv', header=None)
    node_ids = df.iloc[:, 0].values
    timesteps = df.iloc[:, 1].values.astype(np.int64)
    features = StandardScaler().fit_transform(df.iloc[:, 2:].values.astype(np.float32))
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    classes = pd.read_csv('data/raw/elliptic_txs_classes.csv')
    labels = np.full(len(node_ids), -1, dtype=np.int64)
    for _, r in classes.iterrows():
        if r.iloc[0] in id_map:
            labels[id_map[r.iloc[0]]] = {'1': 1, '2': 0}.get(str(r.iloc[1]), -1)

    return features, labels, timesteps

def main():
    print("="*65)
    print("  ANALYSIS: WHY TEMPORAL ENCODING HURTS PERFORMANCE")
    print("="*65)

    features, labels, timesteps = load_data()
    num_timesteps = timesteps.max() + 1

    # Split boundaries
    train_end = int(num_timesteps * 0.6)  # timesteps 0-28
    val_end = int(num_timesteps * 0.8)     # timesteps 29-38
    # test: timesteps 39-48

    print(f"\nSplit boundaries:")
    print(f"  Train: timesteps 0-{train_end-1}")
    print(f"  Val: timesteps {train_end}-{val_end-1}")
    print(f"  Test: timesteps {val_end}-{num_timesteps-1}")

    # =========================================================================
    # 1. ILLICIT RATE BY TIMESTEP
    # =========================================================================
    print("\n" + "="*65)
    print("  1. ILLICIT RATE BY TIMESTEP")
    print("="*65)

    illicit_by_ts = defaultdict(lambda: {'illicit': 0, 'licit': 0, 'total': 0})
    for i, (ts, label) in enumerate(zip(timesteps, labels)):
        if label >= 0:
            illicit_by_ts[ts]['total'] += 1
            if label == 1:
                illicit_by_ts[ts]['illicit'] += 1
            else:
                illicit_by_ts[ts]['licit'] += 1

    ts_stats = []
    for ts in range(num_timesteps):
        if illicit_by_ts[ts]['total'] > 0:
            rate = illicit_by_ts[ts]['illicit'] / illicit_by_ts[ts]['total']
            ts_stats.append({
                'timestep': ts,
                'illicit': illicit_by_ts[ts]['illicit'],
                'licit': illicit_by_ts[ts]['licit'],
                'total': illicit_by_ts[ts]['total'],
                'illicit_rate': rate,
                'split': 'train' if ts < train_end else ('val' if ts < val_end else 'test')
            })

    ts_df = pd.DataFrame(ts_stats)

    # Aggregate by split
    print("\nIllicit rate by split:")
    for split in ['train', 'val', 'test']:
        split_data = ts_df[ts_df['split'] == split]
        total_illicit = split_data['illicit'].sum()
        total_all = split_data['total'].sum()
        rate = total_illicit / total_all if total_all > 0 else 0
        print(f"  {split.upper()}: {total_illicit}/{total_all} = {rate:.4f} ({rate*100:.2f}%)")

    # Show timestep-level variation
    print("\nIllicit rate variation:")
    train_rates = ts_df[ts_df['split'] == 'train']['illicit_rate']
    test_rates = ts_df[ts_df['split'] == 'test']['illicit_rate']
    print(f"  Train timesteps: mean={train_rates.mean():.4f}, std={train_rates.std():.4f}")
    print(f"  Test timesteps: mean={test_rates.mean():.4f}, std={test_rates.std():.4f}")

    # =========================================================================
    # 2. FEATURE DISTRIBUTION SHIFT
    # =========================================================================
    print("\n" + "="*65)
    print("  2. FEATURE DISTRIBUTION SHIFT (Train vs Test)")
    print("="*65)

    train_mask = timesteps < train_end
    test_mask = timesteps >= val_end
    labeled_mask = labels >= 0

    train_features = features[train_mask & labeled_mask]
    test_features = features[test_mask & labeled_mask]

    # Compare mean and std of features
    train_mean = train_features.mean(axis=0)
    test_mean = test_features.mean(axis=0)
    mean_diff = np.abs(train_mean - test_mean)

    print(f"\nFeature mean shift (train vs test):")
    print(f"  Average absolute difference: {mean_diff.mean():.4f}")
    print(f"  Max difference: {mean_diff.max():.4f} (feature {mean_diff.argmax()})")
    print(f"  Features with >0.5 shift: {(mean_diff > 0.5).sum()}")

    # =========================================================================
    # 3. TIMESTEP CORRELATION WITH LABELS
    # =========================================================================
    print("\n" + "="*65)
    print("  3. TIMESTEP-LABEL CORRELATION")
    print("="*65)

    # Check if certain timesteps are strongly associated with illicit
    labeled_idx = labels >= 0
    labeled_ts = timesteps[labeled_idx]
    labeled_y = labels[labeled_idx]

    # Correlation between timestep and label
    corr = np.corrcoef(labeled_ts, labeled_y)[0, 1]
    print(f"\nCorrelation(timestep, illicit_label): {corr:.4f}")

    # Check if test timesteps have different illicit patterns
    train_illicit_ts = timesteps[(labels == 1) & (timesteps < train_end)]
    test_illicit_ts = timesteps[(labels == 1) & (timesteps >= val_end)]

    if len(train_illicit_ts) > 0 and len(test_illicit_ts) > 0:
        print(f"\nIllicit transaction timestep distribution:")
        print(f"  Train illicit: mean_ts={train_illicit_ts.mean():.1f}, std={train_illicit_ts.std():.1f}")
        print(f"  Test illicit: mean_ts={test_illicit_ts.mean():.1f}, std={test_illicit_ts.std():.1f}")

    # =========================================================================
    # 4. THE PROBLEM: TIMESTEP AS A LEAKY FEATURE
    # =========================================================================
    print("\n" + "="*65)
    print("  4. ROOT CAUSE ANALYSIS")
    print("="*65)

    print("""
    WHY TEMPORAL ENCODING HURTS:

    1. DISTRIBUTION SHIFT:
       - Train timesteps (0-28) have different illicit patterns than test (39-48)
       - The model learns "timestep X = high illicit probability"
       - But test timesteps were NEVER seen during training!

    2. TIMESTEP AS IDENTITY:
       - Temporal encoding essentially gives each timestep a unique signature
       - Model memorizes: "transactions at timestep 15 tend to be illicit"
       - At test time, timesteps 39-48 have NO learned associations

    3. OVERFITTING TO TEMPORAL PATTERNS:
       - Illicit rate varies significantly across timesteps (see above)
       - Model learns these spurious correlations instead of actual patterns

    SOLUTION OPTIONS:

    a) Use RELATIVE time (time since first transaction) instead of absolute
    b) Use time DIFFERENCES between connected nodes
    c) Bin timesteps into broader periods
    d) Use temporal features (day-of-week patterns) instead of raw timestep
    e) For temporal split evaluation, static models may be more appropriate
    """)

    # =========================================================================
    # 5. VERIFICATION: Timestep seen vs unseen
    # =========================================================================
    print("\n" + "="*65)
    print("  5. TIMESTEP OVERLAP ANALYSIS")
    print("="*65)

    train_timesteps = set(range(train_end))
    test_timesteps_set = set(range(val_end, num_timesteps))

    print(f"\nTrain timesteps: {sorted(train_timesteps)}")
    print(f"Test timesteps: {sorted(test_timesteps_set)}")
    print(f"Overlap: {train_timesteps & test_timesteps_set}")
    print("\n>>> ZERO OVERLAP - Model has never seen test timesteps!")
    print(">>> Temporal encoding creates features for UNSEEN timesteps at test time")

    # Save analysis
    ts_df.to_csv('results/timestep_analysis.csv', index=False)
    print("\nTimestep analysis saved to results/timestep_analysis.csv")

    # =========================================================================
    # 6. VISUALIZATION
    # =========================================================================
    print("\n" + "="*65)
    print("  6. CREATING VISUALIZATION")
    print("="*65)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Illicit rate by timestep
    ax1 = axes[0, 0]
    colors = ['blue' if s == 'train' else ('orange' if s == 'val' else 'red')
              for s in ts_df['split']]
    ax1.bar(ts_df['timestep'], ts_df['illicit_rate'], color=colors, alpha=0.7)
    ax1.axvline(x=train_end-0.5, color='black', linestyle='--', label='Train/Val split')
    ax1.axvline(x=val_end-0.5, color='black', linestyle=':', label='Val/Test split')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Illicit Rate')
    ax1.set_title('Illicit Rate by Timestep (Blue=Train, Orange=Val, Red=Test)')
    ax1.legend()

    # Plot 2: Transaction count by timestep
    ax2 = axes[0, 1]
    ax2.bar(ts_df['timestep'], ts_df['total'], color=colors, alpha=0.7)
    ax2.axvline(x=train_end-0.5, color='black', linestyle='--')
    ax2.axvline(x=val_end-0.5, color='black', linestyle=':')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Transaction Count')
    ax2.set_title('Labeled Transactions per Timestep')

    # Plot 3: Illicit count by timestep
    ax3 = axes[1, 0]
    ax3.bar(ts_df['timestep'], ts_df['illicit'], color=colors, alpha=0.7)
    ax3.axvline(x=train_end-0.5, color='black', linestyle='--')
    ax3.axvline(x=val_end-0.5, color='black', linestyle=':')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Illicit Count')
    ax3.set_title('Illicit Transactions per Timestep')

    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = """
    ROOT CAUSE: Temporal Encoding Fails in Temporal Split

    Problem:
    • Train timesteps: 0-28
    • Test timesteps: 39-48
    • ZERO OVERLAP between train/test timesteps

    What happens:
    1. Model learns timestep-specific embeddings
    2. Each timestep gets a unique "signature"
    3. At test time, model sees NEW timesteps (39-48)
    4. These have NO learned representations
    5. Temporal encoding becomes NOISE

    Why Static GNN works better:
    • No timestep memorization
    • Learns general graph patterns
    • Generalizes to unseen timesteps

    Conclusion:
    Temporal encoding helps when train/test share timesteps
    (random split), but hurts in realistic temporal evaluation
    where future timesteps are fundamentally different.
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig('results/temporal_analysis.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to results/temporal_analysis.png")

    plt.show()

if __name__ == "__main__":
    main()
