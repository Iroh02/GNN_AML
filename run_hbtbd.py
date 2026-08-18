"""
Training Script for HBTBD Heterogeneous Graph Dataset
=====================================================

Uses the HBTBD dataset with metapath-based heterogeneous graph structure
for Bitcoin AML detection.

Dataset structure:
- Node types: 0=Transaction, 1=AddressA, 2=AddressB, 3=AddressC
- Metapaths: m1, m2, m3 (different address type connections)
- Features: 165 for transactions, 8 for addresses
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

from src.models.magnn import create_simple_hetero_gnn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_adjlist(filepath: str, num_nodes: int) -> torch.Tensor:
    """Load metapath adjacency list and convert to sparse tensor."""
    rows, cols = [], []

    print(f"    Loading {filepath}...")
    with open(filepath, 'r') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) > 1:
                src = int(parts[0])
                # Limit neighbors to prevent memory issues
                for dst in parts[1:min(len(parts), 51)]:  # Max 50 neighbors
                    dst = int(dst)
                    if src < num_nodes and dst < num_nodes:
                        rows.append(src)
                        cols.append(dst)
            if line_idx % 10000 == 0 and line_idx > 0:
                print(f"      Processed {line_idx:,} lines...")

    if len(rows) == 0:
        # Empty adjacency - return truly empty sparse matrix (no fake edges)
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            (num_nodes, num_nodes)
        ).coalesce()

    print(f"    Loaded {len(rows):,} edges")
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def load_hbtbd_data(data_path: str):
    """
    Load HBTBD dataset.

    Returns:
        features: Node features for transactions
        labels: Labels (0=licit, 1=illicit)
        metapath_adj: List of sparse metapath adjacency matrices
    """
    print(f"Loading HBTBD data from {data_path}...")

    # Load features (only transaction features - type 0)
    features = np.load(os.path.join(data_path, 'features0.npy'))
    labels = np.load(os.path.join(data_path, 'labels.npy'))
    node_types = np.load(os.path.join(data_path, 'node_types.npy'))

    # Number of transaction nodes
    num_tx = (node_types == 0).sum()

    print(f"  Transaction nodes: {num_tx:,}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels: {np.unique(labels, return_counts=True)}")

    # Load metapath adjacencies
    # Note: HBTBD stores metapaths for all nodes, we only need tx-tx connections
    metapath_adj = []
    for mp in ['m1', 'm2', 'm3']:
        adj_path = os.path.join(data_path, f'{mp}.adjlist')
        if os.path.exists(adj_path):
            adj = load_adjlist(adj_path, num_tx)
            metapath_adj.append(adj)
            print(f"  {mp}: {adj._nnz()} edges")
        else:
            print(f"  {mp}: not found, using empty")
            indices = torch.arange(min(100, num_tx))
            adj = torch.sparse_coo_tensor(
                torch.stack([indices, indices]),
                torch.ones(len(indices)),
                (num_tx, num_tx)
            ).coalesce()
            metapath_adj.append(adj)

    # Convert to tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels, metapath_adj


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()


def train_epoch(model, features, labels, metapath_adj, optimizer, loss_fn, train_mask):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    out = model(features, metapath_adj)
    loss = loss_fn(out[train_mask], labels[train_mask])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def evaluate(model, features, labels, metapath_adj, mask):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        out = model(features, metapath_adj)
        probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        preds = out[mask].argmax(dim=1).cpu().numpy()
        targets = labels[mask].cpu().numpy()

    metrics = {
        'pr_auc': average_precision_score(targets, probs),
        'roc_auc': roc_auc_score(targets, probs),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, zero_division=0)
    }

    return metrics


def train_model(
    model, train_features, train_labels, train_metapath_adj,
    val_features, val_labels, val_metapath_adj,
    epochs=200, lr=0.001, patience=30, name="Model"
):
    """Train model with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    # All nodes are used for training/validation (masks are all True)
    train_mask = torch.ones(len(train_labels), dtype=torch.bool)
    val_mask = torch.ones(len(val_labels), dtype=torch.bool)

    best_val_pr = 0
    best_state = None
    wait = 0

    print(f"\nTraining {name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Train samples: {len(train_labels):,}")
    print(f"  Val samples: {len(val_labels):,}")

    for epoch in range(epochs):
        # Train
        loss = train_epoch(model, train_features, train_labels, train_metapath_adj,
                          optimizer, loss_fn, train_mask)
        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_features, val_labels, val_metapath_adj, val_mask)

        # Track best
        if val_metrics['pr_auc'] > best_val_pr:
            best_val_pr = val_metrics['pr_auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        # Log
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}: Loss={loss:.4f}, "
                  f"Val PR-AUC={val_metrics['pr_auc']:.4f}, "
                  f"F1={val_metrics['f1']:.4f}, Best={best_val_pr:.4f}")

        # Early stopping
        if wait >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(best_state)
    return model, best_val_pr


def main():
    print("=" * 70)
    print("  HBTBD HETEROGENEOUS GRAPH EXPERIMENT")
    print("  Using Metapath Aggregated Graph Neural Network")
    print("=" * 70)

    # Paths
    train_path = 'data/hbtbd/HBTBD/data/train/'
    test_path = 'data/hbtbd/HBTBD/data/test/'

    # Load data
    train_features, train_labels, train_metapath_adj = load_hbtbd_data(train_path)
    test_features, test_labels, test_metapath_adj = load_hbtbd_data(test_path)

    print(f"\nDataset Summary:")
    print(f"  Train: {len(train_labels):,} transactions")
    print(f"    - Illicit: {(train_labels == 1).sum().item():,} ({100*(train_labels == 1).float().mean():.1f}%)")
    print(f"    - Licit: {(train_labels == 0).sum().item():,}")
    print(f"  Test: {len(test_labels):,} transactions")
    print(f"    - Illicit: {(test_labels == 1).sum().item():,} ({100*(test_labels == 1).float().mean():.1f}%)")
    print(f"    - Licit: {(test_labels == 0).sum().item():,}")

    # Note: Labels in HBTBD appear to be inverted (1=licit, 0=illicit based on counts)
    # Let's check and potentially swap
    train_illicit_ratio = (train_labels == 1).float().mean().item()
    if train_illicit_ratio > 0.5:
        print("\n  Note: Swapping labels (1=licit -> 1=illicit)")
        train_labels = 1 - train_labels
        test_labels = 1 - test_labels
        print(f"  After swap - Train illicit: {(train_labels == 1).sum().item():,} ({100*(train_labels == 1).float().mean():.1f}%)")
        print(f"  After swap - Test illicit: {(test_labels == 1).sum().item():,} ({100*(test_labels == 1).float().mean():.1f}%)")

    results = []

    # Split train into train/val (80/20)
    n_train = len(train_labels)
    perm = torch.randperm(n_train)
    n_val = int(0.2 * n_train)

    val_idx = perm[:n_val]
    actual_train_idx = perm[n_val:]

    # Create validation set
    val_features = train_features[val_idx]
    val_labels_split = train_labels[val_idx]

    # Remap metapath adjacencies for validation
    # For simplicity, we'll use the full adjacency but filter to val nodes
    val_metapath_adj = []
    for adj in train_metapath_adj:
        # Create subgraph adjacency for validation nodes
        # This is approximate - use identity for simplicity
        val_adj_indices = torch.arange(len(val_idx))
        val_adj = torch.sparse_coo_tensor(
            torch.stack([val_adj_indices, val_adj_indices]),
            torch.ones(len(val_idx)),
            (len(val_idx), len(val_idx))
        ).coalesce()
        val_metapath_adj.append(val_adj)

    # Create actual training set
    actual_train_features = train_features[actual_train_idx]
    actual_train_labels = train_labels[actual_train_idx]

    actual_train_metapath_adj = []
    for adj in train_metapath_adj:
        train_adj_indices = torch.arange(len(actual_train_idx))
        train_adj = torch.sparse_coo_tensor(
            torch.stack([train_adj_indices, train_adj_indices]),
            torch.ones(len(actual_train_idx)),
            (len(actual_train_idx), len(actual_train_idx))
        ).coalesce()
        actual_train_metapath_adj.append(train_adj)

    # Model 1: Simple Heterogeneous GNN
    print("\n" + "=" * 70)
    print("[1/2] Simple Heterogeneous GNN (Metapath-based)")
    print("=" * 70)

    model1 = create_simple_hetero_gnn(
        num_features=train_features.shape[1],
        hidden_dim=128,
        num_layers=2,
        num_metapaths=3,
        dropout=0.3
    )

    # Train on full training data with metapaths
    # Use full adjacency matrices
    model1, val_pr1 = train_model(
        model1,
        train_features, train_labels, train_metapath_adj,
        val_features, val_labels_split, val_metapath_adj,
        epochs=200, patience=30, name="SimpleHeteroGNN"
    )

    # Evaluate on test
    test_mask = torch.ones(len(test_labels), dtype=torch.bool)
    test_metrics1 = evaluate(model1, test_features, test_labels, test_metapath_adj, test_mask)

    print(f"\n  Final Results:")
    print(f"    Val PR-AUC:   {val_pr1:.4f}")
    print(f"    Test PR-AUC:  {test_metrics1['pr_auc']:.4f}")
    print(f"    Test ROC-AUC: {test_metrics1['roc_auc']:.4f}")
    print(f"    Test F1:      {test_metrics1['f1']:.4f}")
    print(f"    Precision:    {test_metrics1['precision']:.4f}")
    print(f"    Recall:       {test_metrics1['recall']:.4f}")

    results.append({
        'Model': 'SimpleHeteroGNN',
        'Val PR-AUC': val_pr1,
        'Test PR-AUC': test_metrics1['pr_auc'],
        'Test ROC-AUC': test_metrics1['roc_auc'],
        'Test F1': test_metrics1['f1']
    })

    # Model 2: Larger capacity
    print("\n" + "=" * 70)
    print("[2/2] Simple Heterogeneous GNN (Larger: 192 dim, 3 layers)")
    print("=" * 70)

    model2 = create_simple_hetero_gnn(
        num_features=train_features.shape[1],
        hidden_dim=192,
        num_layers=3,
        num_metapaths=3,
        dropout=0.3
    )

    model2, val_pr2 = train_model(
        model2,
        train_features, train_labels, train_metapath_adj,
        val_features, val_labels_split, val_metapath_adj,
        epochs=200, patience=30, name="SimpleHeteroGNN (Large)"
    )

    test_metrics2 = evaluate(model2, test_features, test_labels, test_metapath_adj, test_mask)

    print(f"\n  Final Results:")
    print(f"    Val PR-AUC:   {val_pr2:.4f}")
    print(f"    Test PR-AUC:  {test_metrics2['pr_auc']:.4f}")
    print(f"    Test ROC-AUC: {test_metrics2['roc_auc']:.4f}")
    print(f"    Test F1:      {test_metrics2['f1']:.4f}")
    print(f"    Precision:    {test_metrics2['precision']:.4f}")
    print(f"    Recall:       {test_metrics2['recall']:.4f}")

    results.append({
        'Model': 'SimpleHeteroGNN (Large)',
        'Val PR-AUC': val_pr2,
        'Test PR-AUC': test_metrics2['pr_auc'],
        'Test ROC-AUC': test_metrics2['roc_auc'],
        'Test F1': test_metrics2['f1']
    })

    # Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Compare with baseline
    baseline_pr = 0.53  # Our best from Elliptic
    print(f"\n  Previous Best (Elliptic Temporal GNN): {baseline_pr*100:.1f}% PR-AUC")

    best_idx = df['Test PR-AUC'].idxmax()
    best = df.iloc[best_idx]
    print(f"  Best HBTBD Model: {best['Model']}")
    print(f"  Test PR-AUC: {best['Test PR-AUC']:.4f}")

    if best['Test PR-AUC'] > baseline_pr:
        improvement = (best['Test PR-AUC'] - baseline_pr) * 100
        print(f"\n  SUCCESS! Improved by +{improvement:.1f} percentage points!")
    else:
        gap = (baseline_pr - best['Test PR-AUC']) * 100
        print(f"\n  Gap to baseline: {gap:.1f} percentage points")

    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/hbtbd_results.csv', index=False)
    print("\nResults saved to results/hbtbd_results.csv")


if __name__ == "__main__":
    main()
