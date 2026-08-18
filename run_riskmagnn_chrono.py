"""
RiskMAGNN on Chronological Resplit
===================================

Train: timesteps 1-40  (includes M2 edges from ts 35-40)
Val:   timesteps 41-45
Test:  timesteps 46-49

This is a temporally-valid split that also includes M2 in training.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.models.riskmagnn import RiskMAGNN, create_riskmagnn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_adjlist(filepath: str, num_nodes: int) -> torch.Tensor:
    """Load metapath adjacency list and convert to sparse tensor."""
    rows, cols = [], []

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
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
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            (num_nodes, num_nodes)
        ).coalesce()

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def load_hbtbd_data(data_path: str, normalize: bool = True, scaler=None):
    """Load HBTBD dataset with optional normalization."""
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


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]

        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        probs = F.softmax(logits, dim=1)
        pt = (probs * smooth_targets).sum(dim=1)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        ce = -torch.sum(smooth_targets * F.log_softmax(logits, dim=1), dim=1)

        return (focal_weight * ce).mean()


def train_epoch(model, features, labels, metapath_adj, optimizer, loss_fn):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    out = model(features, metapath_adj)
    loss = loss_fn(out, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def evaluate(model, features, labels, metapath_adj):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        out = model(features, metapath_adj)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

    if len(np.unique(targets)) < 2:
        return {'pr_auc': 0.0, 'roc_auc': 0.0, 'precision': 0.0,
                'recall': 0.0, 'f1': 0.0}

    return {
        'pr_auc': average_precision_score(targets, probs),
        'roc_auc': roc_auc_score(targets, probs),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, zero_division=0)
    }


def train_model(
    model, train_features, train_labels, train_metapath_adj,
    val_features, val_labels, val_metapath_adj,
    epochs=300, lr=0.001, patience=40, name="RiskMAGNN"
):
    """Train RiskMAGNN with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    warmup_epochs = 15
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75, smoothing=0.1)

    best_val_pr = 0
    best_state = None
    wait = 0

    print(f"\nTraining {name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Train: {len(train_labels):,} nodes")
    print(f"  Val: {len(val_labels):,} nodes")

    for epoch in range(epochs):
        loss = train_epoch(model, train_features, train_labels, train_metapath_adj,
                          optimizer, loss_fn)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, val_features, val_labels, val_metapath_adj)

            if val_metrics['pr_auc'] > best_val_pr:
                best_val_pr = val_metrics['pr_auc']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if (epoch + 1) % 20 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch + 1:3d}: Loss={loss:.4f}, LR={lr_now:.6f}, "
                      f"Val PR-AUC={val_metrics['pr_auc']:.4f}, Best={best_val_pr:.4f}")

            if wait >= patience // 10:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_pr


def main():
    print("=" * 70)
    print("  RiskMAGNN: Chronological Resplit (Train 1-40, Val 41-45, Test 46-49)")
    print("  Temporally valid split WITH M2 edges in training")
    print("=" * 70)

    train_path = 'data/hbtbd_chrono/train/'
    val_path = 'data/hbtbd_chrono/val/'
    test_path = 'data/hbtbd_chrono/test/'

    # Load data (fit scaler on train only)
    print("\n[1] Loading data...")
    train_features, train_labels, train_ts, train_metapath_adj, scaler = \
        load_hbtbd_data(train_path, normalize=True)

    val_features, val_labels, val_ts, val_metapath_adj, _ = \
        load_hbtbd_data(val_path, normalize=True, scaler=scaler)

    test_features, test_labels, test_ts, test_metapath_adj, _ = \
        load_hbtbd_data(test_path, normalize=True, scaler=scaler)

    # Swap labels if needed (illicit is majority in HBTBD)
    if (train_labels == 1).float().mean() > 0.5:
        print("  Swapping labels...")
        train_labels = 1 - train_labels
        val_labels = 1 - val_labels
        test_labels = 1 - test_labels

    print(f"  Train: {len(train_labels):,} nodes (ts {train_ts.min()}-{train_ts.max()}, "
          f"{(train_labels == 1).float().mean()*100:.1f}% illicit)")
    print(f"  Val:   {len(val_labels):,} nodes (ts {val_ts.min()}-{val_ts.max()}, "
          f"{(val_labels == 1).float().mean()*100:.1f}% illicit)")
    print(f"  Test:  {len(test_labels):,} nodes (ts {test_ts.min()}-{test_ts.max()}, "
          f"{(test_labels == 1).float().mean()*100:.1f}% illicit)")

    # Print M2 edge counts
    for name, adj_list in [("Train", train_metapath_adj), ("Val", val_metapath_adj), ("Test", test_metapath_adj)]:
        m2_edges = adj_list[1]._indices().shape[1] if adj_list[1]._indices().shape[1] > 0 else 0
        print(f"  {name} M2 edges: {m2_edges}")

    results = []

    # Model 1: RiskMAGNN (Base)
    print("\n" + "=" * 70)
    print("[2] RiskMAGNN (Base)")
    print("=" * 70)

    model1 = create_riskmagnn(
        num_features=train_features.shape[1],
        hidden_dim=128,
        num_layers=2,
        num_metapaths=3,
        dropout=0.35
    )

    model1, val_pr1 = train_model(
        model1,
        train_features, train_labels, train_metapath_adj,
        val_features, val_labels, val_metapath_adj,
        epochs=350, patience=50, name="RiskMAGNN (Base)"
    )

    test_metrics1 = evaluate(model1, test_features, test_labels, test_metapath_adj)

    print(f"\n  Results:")
    print(f"    Val PR-AUC:   {val_pr1:.4f}")
    print(f"    Test PR-AUC:  {test_metrics1['pr_auc']:.4f}")
    print(f"    Test ROC-AUC: {test_metrics1['roc_auc']:.4f}")
    print(f"    Test F1:      {test_metrics1['f1']:.4f}")
    print(f"    Precision:    {test_metrics1['precision']:.4f}")
    print(f"    Recall:       {test_metrics1['recall']:.4f}")

    attn_weights = model1.get_attention_weights(test_features, test_metapath_adj)
    print(f"\n  Metapath Attention (layer 0 mean):")
    print(f"    M1 (A-centralized): {attn_weights[0][:, 0].mean():.3f}")
    print(f"    M2 (B-one-way):     {attn_weights[0][:, 1].mean():.3f}")
    print(f"    M3 (C-decentralized): {attn_weights[0][:, 2].mean():.3f}")

    results.append({
        'Model': 'RiskMAGNN (Base)',
        'Val PR-AUC': val_pr1,
        'Test PR-AUC': test_metrics1['pr_auc'],
        'Test ROC-AUC': test_metrics1['roc_auc'],
        'Test F1': test_metrics1['f1']
    })

    # Model 2: RiskMAGNN (Large)
    print("\n" + "=" * 70)
    print("[3] RiskMAGNN (Large)")
    print("=" * 70)

    model2 = create_riskmagnn(
        num_features=train_features.shape[1],
        hidden_dim=192,
        num_layers=3,
        num_metapaths=3,
        dropout=0.4
    )

    model2, val_pr2 = train_model(
        model2,
        train_features, train_labels, train_metapath_adj,
        val_features, val_labels, val_metapath_adj,
        epochs=400, patience=60, name="RiskMAGNN (Large)"
    )

    test_metrics2 = evaluate(model2, test_features, test_labels, test_metapath_adj)

    print(f"\n  Results:")
    print(f"    Val PR-AUC:   {val_pr2:.4f}")
    print(f"    Test PR-AUC:  {test_metrics2['pr_auc']:.4f}")
    print(f"    Test ROC-AUC: {test_metrics2['roc_auc']:.4f}")
    print(f"    Test F1:      {test_metrics2['f1']:.4f}")

    results.append({
        'Model': 'RiskMAGNN (Large)',
        'Val PR-AUC': val_pr2,
        'Test PR-AUC': test_metrics2['pr_auc'],
        'Test ROC-AUC': test_metrics2['roc_auc'],
        'Test F1': test_metrics2['f1']
    })

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY - Chronological Resplit")
    print("=" * 70)

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    best_pr = max(test_metrics1['pr_auc'], test_metrics2['pr_auc'])
    print(f"\n  Official split best: 69.02% PR-AUC")
    print(f"  Random resplit best: 94.20% PR-AUC")
    print(f"  Chrono resplit best: {best_pr*100:.2f}% PR-AUC")

    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/riskmagnn_chrono_results.csv', index=False)
    print("\nResults saved to results/riskmagnn_chrono_results.csv")

    # Save best model
    os.makedirs('models', exist_ok=True)
    best_model = model1 if test_metrics1['pr_auc'] > test_metrics2['pr_auc'] else model2
    torch.save(best_model.state_dict(), 'models/riskmagnn_chrono.pth')
    print("Best model saved to models/riskmagnn_chrono.pth")


if __name__ == "__main__":
    main()
