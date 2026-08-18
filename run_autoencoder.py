"""
GNN Autoencoder Training for AML Detection
===========================================

Training strategy:
1. Joint training: reconstruction + classification
2. Reconstruction error as anomaly signal
3. Pre-training option on all nodes (including unlabeled)
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
from torch_geometric.utils import negative_sampling
import warnings
warnings.filterwarnings('ignore')

from src.models.gnn_autoencoder import create_gnn_autoencoder

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


class FocalLoss(nn.Module):
    """Focal loss for classification."""
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


class OriginalTemporalGNN(nn.Module):
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
            h = F.dropout(F.relu(b(c(h, ei))), self.drop, self.training)
            if i == 0:
                h = h + h_skip
            else:
                h = h + h_in
        return self.clf(h)


def pretrain_autoencoder(model, data, epochs=50, lr=0.001):
    """
    Pre-train autoencoder on ALL nodes (including unlabeled).
    Only uses reconstruction loss, no classification.
    """
    print("\n  Pre-training autoencoder on all nodes...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(data.x, data.edge_index, data.timestep)

        # Reconstruction loss on ALL nodes
        recon_loss = model.reconstruction_loss(data.x, output['x_recon'])

        # Edge prediction loss
        neg_edge_index = negative_sampling(
            data.edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=data.edge_index.size(1) // 2
        )
        edge_loss = model.edge_loss(output['embeddings'], data.edge_index, neg_edge_index)

        loss = recon_loss + 0.5 * edge_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Pretrain Epoch {epoch + 1}: Recon={recon_loss.item():.4f}, Edge={edge_loss.item():.4f}")

    print("  Pre-training complete.")


def train_autoencoder(model, data, train_mask, val_mask, test_mask,
                      epochs=300, lr=0.001, patience=50, pretrain=True,
                      recon_weight=0.1, name="Model"):
    """
    Train GNN Autoencoder with joint reconstruction + classification.
    """
    # Optional pre-training
    if pretrain:
        pretrain_autoencoder(model, data, epochs=50, lr=lr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    cls_loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    best_val_pr = 0
    best_state = None
    wait = 0

    print(f"\n  Training {name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(data.x, data.edge_index, data.timestep)

        # Classification loss (on labeled training nodes only)
        cls_loss = cls_loss_fn(output['logits'][train_mask], data.y[train_mask])

        # Reconstruction loss (on ALL nodes - semi-supervised signal)
        recon_loss = model.reconstruction_loss(data.x, output['x_recon'])

        # Combined loss
        loss = cls_loss + recon_weight * recon_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            output = model(data.x, data.edge_index, data.timestep)
            val_probs = F.softmax(output['logits'][val_mask], dim=1)[:, 1]
            val_pr = average_precision_score(data.y[val_mask].numpy(), val_probs.numpy())
            val_roc = roc_auc_score(data.y[val_mask].numpy(), val_probs.numpy())

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1}: Loss={loss.item():.4f}, Cls={cls_loss.item():.4f}, "
                  f"Recon={recon_loss.item():.4f}, Val PR={val_pr:.4f}, Best={best_val_pr:.4f}")

        if wait >= patience:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    # Load best and evaluate
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        output = model(data.x, data.edge_index, data.timestep)

        test_probs = F.softmax(output['logits'][test_mask], dim=1)[:, 1]
        test_pr = average_precision_score(data.y[test_mask].numpy(), test_probs.numpy())
        test_roc = roc_auc_score(data.y[test_mask].numpy(), test_probs.numpy())

        val_probs = F.softmax(output['logits'][val_mask], dim=1)[:, 1]
        val_pr = average_precision_score(data.y[val_mask].numpy(), val_probs.numpy())

        # Also check reconstruction error as standalone predictor
        recon_error = output['recon_error']
        recon_pr = average_precision_score(data.y[test_mask].numpy(), recon_error[test_mask].numpy())

    print(f"\n  Final Results for {name}:")
    print(f"    Val PR-AUC:  {val_pr:.4f}")
    print(f"    Test PR-AUC: {test_pr:.4f}")
    print(f"    Test ROC-AUC: {test_roc:.4f}")
    print(f"    Recon Error as Predictor: {recon_pr:.4f}")

    return test_pr, test_roc, val_pr, recon_pr


def train_baseline(model, data, train_mask, val_mask, test_mask,
                   epochs=300, lr=0.001, patience=50, name="Model"):
    """Train baseline model without autoencoder."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    best_val_pr = 0
    best_state = None
    wait = 0

    print(f"\n  Training {name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.timestep)
        loss = loss_fn(out[train_mask], data.y[train_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.timestep)
            val_probs = F.softmax(out[val_mask], dim=1)[:, 1]
            val_pr = average_precision_score(data.y[val_mask].numpy(), val_probs.numpy())

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1}: Loss={loss.item():.4f}, Val PR={val_pr:.4f}, Best={best_val_pr:.4f}")

        if wait >= patience:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.timestep)
        test_probs = F.softmax(out[test_mask], dim=1)[:, 1]
        test_pr = average_precision_score(data.y[test_mask].numpy(), test_probs.numpy())
        test_roc = roc_auc_score(data.y[test_mask].numpy(), test_probs.numpy())
        val_probs = F.softmax(out[val_mask], dim=1)[:, 1]
        val_pr = average_precision_score(data.y[val_mask].numpy(), val_probs.numpy())

    print(f"\n  Final Results for {name}:")
    print(f"    Val PR-AUC:  {val_pr:.4f}")
    print(f"    Test PR-AUC: {test_pr:.4f}")
    print(f"    Test ROC-AUC: {test_roc:.4f}")

    return test_pr, test_roc, val_pr


def main():
    print("=" * 70)
    print("  GNN AUTOENCODER EXPERIMENT")
    print("  Goal: Use reconstruction error as anomaly signal")
    print("=" * 70)

    data = load_data()
    train_mask, val_mask, test_mask = get_masks(data)

    print(f"\nDataset splits:")
    print(f"  Train: {train_mask.sum().item():,} labeled")
    print(f"  Val:   {val_mask.sum().item():,} labeled")
    print(f"  Test:  {test_mask.sum().item():,} labeled")
    print(f"  Unlabeled: {(data.y == -1).sum().item():,} (used for reconstruction)")

    results = []

    # 1. Baseline: Original Temporal GNN
    print("\n" + "=" * 70)
    print("[1/4] Baseline: Original Temporal GNN")
    print("=" * 70)

    baseline = OriginalTemporalGNN(in_dim=data.x.shape[1], hid=128, tdim=32, layers=2)
    base_pr, base_roc, base_val = train_baseline(
        baseline, data, train_mask, val_mask, test_mask, name="Original Temporal GNN"
    )
    results.append({
        'Model': 'Original Temporal GNN',
        'Val PR-AUC': base_val,
        'Test PR-AUC': base_pr,
        'Test ROC-AUC': base_roc,
        'Recon PR-AUC': '-'
    })

    # 2. GNN Autoencoder (no pre-training)
    print("\n" + "=" * 70)
    print("[2/4] GNN Autoencoder (No Pre-training)")
    print("=" * 70)

    ae_model1 = create_gnn_autoencoder(
        num_features=data.x.shape[1],
        hidden_dim=128,
        embed_dim=64,
        num_layers=2,
        use_temporal=True
    )
    ae1_pr, ae1_roc, ae1_val, ae1_recon = train_autoencoder(
        ae_model1, data, train_mask, val_mask, test_mask,
        pretrain=False, recon_weight=0.1, name="GNN Autoencoder (No Pretrain)"
    )
    results.append({
        'Model': 'GNN Autoencoder (No Pretrain)',
        'Val PR-AUC': ae1_val,
        'Test PR-AUC': ae1_pr,
        'Test ROC-AUC': ae1_roc,
        'Recon PR-AUC': ae1_recon
    })

    # 3. GNN Autoencoder (with pre-training)
    print("\n" + "=" * 70)
    print("[3/4] GNN Autoencoder (With Pre-training on All Nodes)")
    print("=" * 70)

    ae_model2 = create_gnn_autoencoder(
        num_features=data.x.shape[1],
        hidden_dim=128,
        embed_dim=64,
        num_layers=2,
        use_temporal=True
    )
    ae2_pr, ae2_roc, ae2_val, ae2_recon = train_autoencoder(
        ae_model2, data, train_mask, val_mask, test_mask,
        pretrain=True, recon_weight=0.1, name="GNN Autoencoder (Pretrained)"
    )
    results.append({
        'Model': 'GNN Autoencoder (Pretrained)',
        'Val PR-AUC': ae2_val,
        'Test PR-AUC': ae2_pr,
        'Test ROC-AUC': ae2_roc,
        'Recon PR-AUC': ae2_recon
    })

    # 4. GNN Autoencoder (larger, more reconstruction weight)
    print("\n" + "=" * 70)
    print("[4/4] GNN Autoencoder (Larger + Higher Recon Weight)")
    print("=" * 70)

    ae_model3 = create_gnn_autoencoder(
        num_features=data.x.shape[1],
        hidden_dim=192,
        embed_dim=96,
        num_layers=3,
        use_temporal=True
    )
    ae3_pr, ae3_roc, ae3_val, ae3_recon = train_autoencoder(
        ae_model3, data, train_mask, val_mask, test_mask,
        pretrain=True, recon_weight=0.3, name="GNN Autoencoder (Large)"
    )
    results.append({
        'Model': 'GNN Autoencoder (Large)',
        'Val PR-AUC': ae3_val,
        'Test PR-AUC': ae3_pr,
        'Test ROC-AUC': ae3_roc,
        'Recon PR-AUC': ae3_recon
    })

    # Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Find best
    numeric_results = df[df['Recon PR-AUC'] != '-'].copy()
    if len(numeric_results) > 0:
        numeric_results['Test PR-AUC'] = numeric_results['Test PR-AUC'].astype(float)
        best_ae_idx = numeric_results['Test PR-AUC'].idxmax()
        best_ae = df.iloc[best_ae_idx]
    else:
        best_ae = None

    print(f"\n{'=' * 70}")
    print(f"  BASELINE: Original Temporal GNN = {base_pr:.4f} Test PR-AUC")
    if best_ae is not None:
        print(f"  BEST AUTOENCODER: {best_ae['Model']} = {best_ae['Test PR-AUC']:.4f} Test PR-AUC")
        improvement = (float(best_ae['Test PR-AUC']) - base_pr) * 100
        print(f"  IMPROVEMENT: {improvement:+.1f} percentage points")
    print("=" * 70)

    # Check target
    all_pr = [base_pr] + [r['Test PR-AUC'] for r in results[1:]]
    best_pr = max(all_pr)

    if best_pr >= 0.60:
        print(f"\nSUCCESS! Achieved {best_pr*100:.1f}% PR-AUC (target: 60%)")
    else:
        gap = (0.60 - best_pr) * 100
        print(f"\nBest: {best_pr*100:.1f}% PR-AUC")
        print(f"Gap to 60% target: {gap:.1f} percentage points")

    # Save
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/autoencoder_results.csv', index=False)
    print("\nResults saved to results/autoencoder_results.csv")


if __name__ == "__main__":
    main()
