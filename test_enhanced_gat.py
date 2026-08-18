"""
Quick test of Enhanced GAT model (single configuration, short training)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

from src.models.enhanced_temporal_gat import create_enhanced_temporal_gat

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    """Load Elliptic dataset."""
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
    train_mask = (data.timestep < int(n*0.6)) & lab
    val_mask = (data.timestep >= int(n*0.6)) & (data.timestep < int(n*0.8)) & lab
    test_mask = (data.timestep >= int(n*0.8)) & lab
    unlabeled_mask = (data.y == -1)

    return train_mask, val_mask, test_mask, unlabeled_mask


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def quick_train():
    """Quick training run to test the model."""
    print("="*70)
    print("QUICK TEST: Enhanced Temporal GAT")
    print("="*70)

    # Load data
    print("\nLoading data...")
    data = load_data()
    train_mask, val_mask, test_mask, unlabeled_mask = get_masks(data)

    print(f"  Train: {train_mask.sum().item()}")
    print(f"  Val: {val_mask.sum().item()}")
    print(f"  Test: {test_mask.sum().item()}")
    print(f"  Unlabeled: {unlabeled_mask.sum().item()}")

    # Create model
    print("\nCreating Enhanced GAT model...")
    model = create_enhanced_temporal_gat(
        num_features=data.x.shape[1],
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        use_behavioral=True
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    loss_fn = FocalLoss()

    print("\nTraining (100 epochs, no SSL for speed)...")
    best_val = 0
    best_state = None

    for epoch in range(100):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.timestep)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validate
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.timestep)
                val_probs = F.softmax(out[val_mask], dim=1)[:, 1]
                val_pr = average_precision_score(data.y[val_mask].numpy(), val_probs.numpy())
                val_roc = roc_auc_score(data.y[val_mask].numpy(), val_probs.numpy())

            if val_pr > best_val:
                best_val = val_pr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Val PR-AUC={val_pr:.4f}, ROC-AUC={val_roc:.4f}")

    # Test with best model
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.timestep)
        test_probs = F.softmax(out[test_mask], dim=1)[:, 1]
        test_pr = average_precision_score(data.y[test_mask].numpy(), test_probs.numpy())
        test_roc = roc_auc_score(data.y[test_mask].numpy(), test_probs.numpy())

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Test PR-AUC: {test_pr:.4f}")
    print(f"Test ROC-AUC: {test_roc:.4f}")

    if test_pr >= 0.60:
        print(f"\nSUCCESS! Achieved {test_pr*100:.1f}% PR-AUC (target: 60%)")
    else:
        gap = (0.60 - test_pr) * 100
        print(f"\nTest: {test_pr*100:.1f}% PR-AUC")
        print(f"Gap to target: {gap:.1f} percentage points")

    print("\nNote: This was a quick test without semi-supervised learning.")
    print("Full training with SSL should achieve higher performance.")


if __name__ == "__main__":
    quick_train()
