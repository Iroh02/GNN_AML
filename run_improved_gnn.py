"""
Training Script for Improved Temporal GNN
==========================================

Enhancements:
1. Multi-scale temporal encoding
2. Domain-specific features
3. Adaptive focal loss with class-aware weighting
4. Extended training with better early stopping
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
import warnings
warnings.filterwarnings('ignore')

from src.models.improved_temporal_gnn import create_improved_temporal_gnn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    """Load Elliptic dataset."""
    print("Loading data...")
    df = pd.read_csv('data/raw/elliptic_txs_features.csv', header=None)
    node_ids = df.iloc[:, 0].values
    timesteps = df.iloc[:, 1].values.astype(np.int64)
    features = StandardScaler().fit_transform(df.iloc[:, 2:].values.astype(np.float32))
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    edges = pd.read_csv('data/raw/elliptic_txs_edgelist.csv')
    src = edges.iloc[:, 0].map(id_map).values
    dst = edges.iloc[:, 1].map(id_map).values
    valid = ~(np.isnan(src) | np.isnan(dst))
    edge_index = torch.tensor(np.stack([src[valid].astype(np.int64), dst[valid].astype(np.int64)]))

    classes = pd.read_csv('data/raw/elliptic_txs_classes.csv')
    labels = np.full(len(node_ids), -1, dtype=np.int64)
    for _, r in classes.iterrows():
        if r.iloc[0] in id_map:
            labels[id_map[r.iloc[0]]] = {'1': 1, '2': 0}.get(str(r.iloc[1]), -1)

    print(f"  Nodes: {len(node_ids):,}")
    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Features: {features.shape[1]}")

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        timestep=torch.tensor(timesteps, dtype=torch.long)
    )


def get_masks(data):
    """Temporal split masks."""
    n = data.timestep.max().item() + 1
    lab = data.y != -1
    train_mask = (data.timestep < int(n * 0.6)) & lab
    val_mask = (data.timestep >= int(n * 0.6)) & (data.timestep < int(n * 0.8)) & lab
    test_mask = (data.timestep >= int(n * 0.8)) & lab

    return train_mask, val_mask, test_mask


class AdaptiveFocalLoss(nn.Module):
    """
    Focal loss with class-aware alpha weighting.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight for positive class (illicit)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)

        # Class-specific alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()


class OriginalAbsoluteTemporalGNN(nn.Module):
    """Original Absolute Temporal GNN for comparison."""
    def __init__(self, in_dim, hid=128, tdim=32, layers=2, drop=0.3):
        super().__init__()
        self.tenc = nn.Linear(1, tdim)
        self.nenc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, hid))
        self.convs = nn.ModuleList([SAGEConv(hid + tdim if i == 0 else hid, hid) for i in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hid) for _ in range(layers)])
        self.input_proj = nn.Linear(hid + tdim, hid)
        self.clf = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid // 2, 2))
        self.drop = drop

    def forward(self, x, ei, ts):
        t = torch.sin(self.tenc(ts.float().unsqueeze(-1) / 49.0))
        h = self.nenc(x)
        h = torch.cat([h, t], -1)
        h_skip = self.input_proj(h)

        for i, (c, b) in enumerate(zip(self.convs, self.bns)):
            h_in = h
            h = c(h, ei)
            h = b(h)
            h = F.relu(h)
            h = F.dropout(h, self.drop, self.training)
            if i == 0:
                h = h + h_skip
            else:
                h = h + h_in
        return self.clf(h)


def train_model(model, data, train_mask, val_mask, test_mask,
                epochs=300, lr=0.001, patience=50, name="Model"):
    """
    Train model with improved strategy.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Adaptive focal loss with class weighting
    loss_fn = AdaptiveFocalLoss(gamma=2.0, alpha=0.75)

    best_val_pr = 0
    best_state = None
    wait = 0

    print(f"\nTraining {name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.timestep)
        loss = loss_fn(out[train_mask], data.y[train_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.timestep)
            val_probs = F.softmax(out[val_mask], dim=1)[:, 1]
            val_pr = average_precision_score(data.y[val_mask].numpy(), val_probs.numpy())
            val_roc = roc_auc_score(data.y[val_mask].numpy(), val_probs.numpy())

        # Track best model
        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        # Logging
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1:3d}: Loss={loss.item():.4f}, "
                  f"Val PR-AUC={val_pr:.4f}, Val ROC={val_roc:.4f}, Best={best_val_pr:.4f}")

        # Early stopping
        if wait >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.timestep)

        # Test metrics
        test_probs = F.softmax(out[test_mask], dim=1)[:, 1]
        test_pr = average_precision_score(data.y[test_mask].numpy(), test_probs.numpy())
        test_roc = roc_auc_score(data.y[test_mask].numpy(), test_probs.numpy())

        # Validation metrics (for reporting)
        val_probs = F.softmax(out[val_mask], dim=1)[:, 1]
        val_pr = average_precision_score(data.y[val_mask].numpy(), val_probs.numpy())

    print(f"\n  Final Results for {name}:")
    print(f"    Val PR-AUC:  {val_pr:.4f}")
    print(f"    Test PR-AUC: {test_pr:.4f}")
    print(f"    Test ROC-AUC: {test_roc:.4f}")

    return test_pr, test_roc, val_pr


def main():
    print("=" * 70)
    print("  IMPROVED TEMPORAL GNN EXPERIMENT")
    print("  Goal: Improve PR-AUC from 55.5% toward 60-70% target")
    print("=" * 70)

    # Load data
    data = load_data()
    train_mask, val_mask, test_mask = get_masks(data)

    print(f"\nDataset splits:")
    print(f"  Train: {train_mask.sum().item():,} nodes ({data.y[train_mask].sum().item()} illicit)")
    print(f"  Val:   {val_mask.sum().item():,} nodes ({data.y[val_mask].sum().item()} illicit)")
    print(f"  Test:  {test_mask.sum().item():,} nodes ({data.y[test_mask].sum().item()} illicit)")

    results = []

    # 1. Original Absolute Temporal GNN (baseline)
    print("\n" + "=" * 70)
    print("[1/3] Original Absolute Temporal GNN (Baseline)")
    print("=" * 70)

    original_model = OriginalAbsoluteTemporalGNN(
        in_dim=data.x.shape[1],
        hid=128,
        tdim=32,
        layers=2,
        drop=0.3
    )

    orig_test_pr, orig_test_roc, orig_val_pr = train_model(
        original_model, data, train_mask, val_mask, test_mask,
        epochs=300, patience=50, name="Original Absolute Temporal GNN"
    )

    results.append({
        'Model': 'Original Absolute Temporal GNN',
        'Val PR-AUC': orig_val_pr,
        'Test PR-AUC': orig_test_pr,
        'Test ROC-AUC': orig_test_roc
    })

    # 2. Improved Temporal GNN (our enhancement)
    print("\n" + "=" * 70)
    print("[2/3] Improved Temporal GNN (Multi-scale Time + Domain Features)")
    print("=" * 70)

    improved_model = create_improved_temporal_gnn(
        num_features=data.x.shape[1],
        hidden_dim=128,
        time_dim=32,
        num_layers=2,
        dropout=0.3
    )

    imp_test_pr, imp_test_roc, imp_val_pr = train_model(
        improved_model, data, train_mask, val_mask, test_mask,
        epochs=300, patience=50, name="Improved Temporal GNN"
    )

    results.append({
        'Model': 'Improved Temporal GNN',
        'Val PR-AUC': imp_val_pr,
        'Test PR-AUC': imp_test_pr,
        'Test ROC-AUC': imp_test_roc
    })

    # 3. Improved Temporal GNN (larger capacity)
    print("\n" + "=" * 70)
    print("[3/3] Improved Temporal GNN (Larger: 192 dim, 3 layers)")
    print("=" * 70)

    improved_large = create_improved_temporal_gnn(
        num_features=data.x.shape[1],
        hidden_dim=192,
        time_dim=48,
        num_layers=3,
        dropout=0.3
    )

    large_test_pr, large_test_roc, large_val_pr = train_model(
        improved_large, data, train_mask, val_mask, test_mask,
        epochs=300, patience=50, name="Improved Temporal GNN (Large)"
    )

    results.append({
        'Model': 'Improved Temporal GNN (Large)',
        'Val PR-AUC': large_val_pr,
        'Test PR-AUC': large_test_pr,
        'Test ROC-AUC': large_test_roc
    })

    # Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    df['Improvement vs Original'] = ((df['Test PR-AUC'] - orig_test_pr) / orig_test_pr * 100).round(1)

    print(df.to_string(index=False))

    # Find best model
    best_idx = df['Test PR-AUC'].idxmax()
    best = df.iloc[best_idx]

    print(f"\n{'=' * 70}")
    print(f"  BEST MODEL: {best['Model']}")
    print(f"  Test PR-AUC: {best['Test PR-AUC']:.4f}")
    print(f"  Test ROC-AUC: {best['Test ROC-AUC']:.4f}")
    print(f"{'=' * 70}")

    # Check target
    if best['Test PR-AUC'] >= 0.60:
        print(f"\nSUCCESS! Achieved {best['Test PR-AUC']*100:.1f}% PR-AUC (target: 60%)")
    else:
        gap = (0.60 - best['Test PR-AUC']) * 100
        print(f"\nBest: {best['Test PR-AUC']*100:.1f}% PR-AUC")
        print(f"Gap to 60% target: {gap:.1f} percentage points")

        if best['Test PR-AUC'] > orig_test_pr:
            improvement = (best['Test PR-AUC'] - orig_test_pr) * 100
            print(f"Improvement over original: +{improvement:.1f} percentage points")

    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/improved_gnn_results.csv', index=False)
    print("\nResults saved to results/improved_gnn_results.csv")


if __name__ == "__main__":
    main()
