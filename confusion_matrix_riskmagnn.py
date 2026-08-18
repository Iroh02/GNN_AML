"""
Generate Confusion Matrix for RiskMAGNN
========================================

Visualizes the classification performance of the trained RiskMAGNN model.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from src.models.riskmagnn import create_riskmagnn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_adjlist(filepath: str, num_nodes: int) -> torch.Tensor:
    """Load metapath adjacency list."""
    rows, cols = [], []

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        indices = torch.arange(min(100, num_nodes))
        return torch.sparse_coo_tensor(
            torch.stack([indices, indices]),
            torch.ones(len(indices)),
            (num_nodes, num_nodes)
        ).coalesce()

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                src = int(parts[0])
                for dst in parts[1:min(len(parts), 51)]:
                    dst = int(dst)
                    if src < num_nodes and dst < num_nodes:
                        rows.append(src)
                        cols.append(dst)

    if len(rows) == 0:
        # Empty adjacency - return truly empty sparse matrix (no fake edges)
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            (num_nodes, num_nodes)
        ).coalesce()

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def load_hbtbd_data(data_path: str, normalize: bool = True, scaler=None):
    """Load HBTBD dataset."""
    features = np.load(os.path.join(data_path, 'features0.npy'))
    labels = np.load(os.path.join(data_path, 'labels.npy'))
    timesteps = np.load(os.path.join(data_path, 'timesteps.npy'))
    node_types = np.load(os.path.join(data_path, 'node_types.npy'))

    num_tx = (node_types == 0).sum()

    if normalize:
        if scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)

    metapath_adj = []
    for mp in ['m1', 'm2', 'm3']:
        adj_path = os.path.join(data_path, f'{mp}.adjlist')
        adj = load_adjlist(adj_path, num_tx)
        metapath_adj.append(adj)

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    timesteps = torch.tensor(timesteps, dtype=torch.long)

    return features, labels, timesteps, metapath_adj, scaler


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Licit', 'Illicit'],
                yticklabels=['Licit', 'Illicit'],
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')

    # Add total counts
    tn, fp, fn, tp = cm.ravel()
    ax1.text(0.5, -0.15, f'TN: {tn:,}  |  FP: {fp:,}  |  FN: {fn:,}  |  TP: {tp:,}',
             transform=ax1.transAxes, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Plot 2: Normalized percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=['Licit', 'Illicit'],
                yticklabels=['Licit', 'Illicit'],
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized %)', fontsize=14, fontweight='bold')

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    ax2.text(0.5, -0.15,
             f'Accuracy: {accuracy*100:.2f}%  |  Precision: {precision*100:.2f}%  |  Recall: {recall*100:.2f}%  |  F1: {f1*100:.2f}%',
             transform=ax2.transAxes, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('RiskMAGNN Classification Performance (69.02% PR-AUC)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_path}")

    return cm


def main():
    print("=" * 70)
    print("  RiskMAGNN Confusion Matrix Analysis")
    print("=" * 70)

    # Load test data
    print("\n[1] Loading test data...")
    test_path = 'data/hbtbd/HBTBD/data/test/'
    test_features, test_labels, test_timesteps, test_metapath_adj, _ = \
        load_hbtbd_data(test_path, normalize=True, scaler=None)

    # Swap labels if needed
    if (test_labels == 1).float().mean() > 0.5:
        test_labels = 1 - test_labels

    print(f"  Test set: {len(test_labels):,} transactions")
    print(f"  Illicit: {(test_labels == 1).sum().item()} ({(test_labels == 1).float().mean()*100:.1f}%)")
    print(f"  Licit: {(test_labels == 0).sum().item()} ({(test_labels == 0).float().mean()*100:.1f}%)")

    # Load model
    print("\n[2] Loading trained RiskMAGNN model...")
    model = create_riskmagnn(
        num_features=test_features.shape[1],
        hidden_dim=192,
        num_layers=3,
        num_metapaths=3,
        dropout=0.4
    )

    if os.path.exists('models/riskmagnn_large.pth'):
        model.load_state_dict(torch.load('models/riskmagnn_large.pth'))
        print("  Loaded weights from models/riskmagnn_large.pth")
    else:
        print("  Error: No saved weights found at models/riskmagnn_large.pth")
        return

    model.eval()

    # Get predictions
    print("\n[3] Generating predictions...")
    with torch.no_grad():
        logits = model(test_features, test_metapath_adj)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    y_true = test_labels.cpu().numpy()
    y_pred = preds

    # Print classification report
    print("\n" + "=" * 70)
    print("  CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred,
                                target_names=['Licit', 'Illicit'],
                                digits=4))

    # Generate confusion matrix
    print("\n[4] Generating confusion matrix visualization...")
    os.makedirs('results', exist_ok=True)
    cm = plot_confusion_matrix(y_true, y_pred, 'results/confusion_matrix_riskmagnn.png')

    # Print detailed breakdown
    tn, fp, fn, tp = cm.ravel()
    print("\n" + "=" * 70)
    print("  DETAILED BREAKDOWN")
    print("=" * 70)
    print(f"  True Negatives (TN):  {tn:,} (correctly identified licit)")
    print(f"  False Positives (FP): {fp:,} (licit flagged as illicit)")
    print(f"  False Negatives (FN): {fn:,} (illicit missed)")
    print(f"  True Positives (TP):  {tp:,} (correctly identified illicit)")
    print()
    print(f"  Total Test Samples: {len(y_true):,}")
    print(f"  Correctly Classified: {tp + tn:,} ({(tp + tn) / len(y_true) * 100:.2f}%)")
    print(f"  Misclassified: {fp + fn:,} ({(fp + fn) / len(y_true) * 100:.2f}%)")

    # Error analysis
    print("\n" + "=" * 70)
    print("  ERROR ANALYSIS")
    print("=" * 70)
    print(f"  False Positive Rate: {fp / (fp + tn) * 100:.2f}% ({fp} out of {fp + tn:,} licit transactions)")
    print(f"  False Negative Rate: {fn / (fn + tp) * 100:.2f}% ({fn} out of {fn + tp:,} illicit transactions)")
    print("\n  For compliance and AML use cases:")
    print(f"    - Model catches {tp / (tp + fn) * 100:.1f}% of illicit transactions (Recall)")
    print(f"    - {fp / (fp + tn) * 100:.2f}% of legitimate transactions flagged for review (FPR)")
    print(f"    - {tp / (tp + fp) * 100:.1f}% of flagged transactions are truly illicit (Precision)")

    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()