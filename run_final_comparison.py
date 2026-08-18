"""
Final Model Comparison
=====================

Compare all models:
1. Logistic Regression (baseline)
2. Static GraphSAGE
3. Original Temporal GNN (absolute timestamps)
4. Relative Temporal GNN (NEW - relative time + behavioral features)

Goal: Achieve 60-70% PR-AUC with Relative Temporal GNN
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
import warnings
warnings.filterwarnings('ignore')

# Import our models
from src.models.relative_temporal_gnn import create_relative_temporal_gnn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data():
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

    return Data(x=torch.tensor(features, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(labels, dtype=torch.long),
                timestep=torch.tensor(timesteps, dtype=torch.long))

def get_masks(data):
    n = data.timestep.max().item() + 1
    lab = data.y != -1
    return ((data.timestep < int(n*0.6)) & lab,
            (data.timestep >= int(n*0.6)) & (data.timestep < int(n*0.8)) & lab,
            (data.timestep >= int(n*0.8)) & lab)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()

# Original models for comparison
class StaticGraphSAGE(nn.Module):
    def __init__(self, in_dim, hid=128, layers=2, drop=0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim if i==0 else hid, hid) for i in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hid) for _ in range(layers)])
        self.clf = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid//2, 2))
        self.drop = drop
    def forward(self, x, ei, ts=None):
        for c, b in zip(self.convs, self.bns):
            x = F.dropout(F.relu(b(c(x, ei))), self.drop, self.training)
        return self.clf(x)

class AbsoluteTemporalGNN(nn.Module):
    """Original temporal GNN with absolute timestamps"""
    def __init__(self, in_dim, hid=128, tdim=32, layers=2, drop=0.3):
        super().__init__()
        self.tenc = nn.Linear(1, tdim)
        self.nenc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.convs = nn.ModuleList([SAGEConv(hid + tdim, hid) for _ in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hid) for _ in range(layers)])
        self.clf = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid//2, 2))
        self.drop = drop
    def forward(self, x, ei, ts):
        t = torch.sin(self.tenc(ts.float().unsqueeze(-1) / 49.0))
        h = self.nenc(x)
        for c, b in zip(self.convs, self.bns):
            h = F.dropout(F.relu(b(c(torch.cat([h, t], -1), ei))), self.drop, self.training) + h
        return self.clf(h)

def train_gnn(model, data, tr, va, te, temporal=False, epochs=250, lr=0.001, name="Model"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
    loss_fn = FocalLoss()
    best, state, wait = 0, None, 0

    print(f"  Training {name}...")
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.timestep) if temporal else model(data.x, data.edge_index)
        loss_fn(out[tr], data.y[tr]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.timestep) if temporal else model(data.x, data.edge_index)
            auc = average_precision_score(data.y[va].numpy(), F.softmax(out[va], 1)[:, 1].numpy())
        if auc > best:
            best, state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if (ep+1) % 50 == 0:
            print(f"    Epoch {ep+1}: Val PR-AUC = {auc:.4f} (Best: {best:.4f})")
        if wait >= 30:
            break

    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.timestep) if temporal else model(data.x, data.edge_index)
        test_auc = average_precision_score(data.y[te].numpy(), F.softmax(out[te], 1)[:, 1].numpy())
    print(f"    Test PR-AUC: {test_auc:.4f}\n")
    return test_auc, best

def main():
    print("="*70)
    print("  FINAL MODEL COMPARISON")
    print("  Goal: Achieve 60-70% PR-AUC with Relative Temporal GNN")
    print("="*70)

    data = load_data()
    tr, va, te = get_masks(data)
    results = []

    # 1. Logistic Regression
    print("\n[1/4] Logistic Regression...")
    X, y = data.x.numpy(), data.y.numpy()
    w = (y[tr.numpy()] == 0).sum() / (y[tr.numpy()] == 1).sum()
    lr = LogisticRegression(class_weight={0:1, 1:w}, C=0.1, max_iter=2000, random_state=SEED)
    lr.fit(X[tr.numpy()], y[tr.numpy()])
    lr_auc = average_precision_score(y[te.numpy()], lr.predict_proba(X[te.numpy()])[:, 1])
    results.append({'Model': 'Logistic Regression', 'Test PR-AUC': lr_auc, 'Type': 'Baseline'})
    print(f"  Test PR-AUC: {lr_auc:.4f}\n")

    # 2. Static GraphSAGE
    print("[2/4] Static GraphSAGE...")
    gs = StaticGraphSAGE(data.x.shape[1], hid=128, layers=2)
    gs_auc, _ = train_gnn(gs, data, tr, va, te, temporal=False, name="GraphSAGE")
    results.append({'Model': 'Static GraphSAGE', 'Test PR-AUC': gs_auc, 'Type': 'Static'})

    # 3. Original Temporal GNN (absolute)
    print("[3/4] Original Temporal GNN (Absolute Timestamps)...")
    abs_tgnn = AbsoluteTemporalGNN(data.x.shape[1], hid=128, tdim=32, layers=2)
    abs_auc, _ = train_gnn(abs_tgnn, data, tr, va, te, temporal=True, name="Absolute TempGNN")
    results.append({'Model': 'Absolute Temporal GNN', 'Test PR-AUC': abs_auc, 'Type': 'Temporal'})

    # 4. NEW: Relative Temporal GNN
    print("[4/4] Relative Temporal GNN (NEW - Relative Time + Behavioral)...")
    rel_tgnn = create_relative_temporal_gnn(
        num_features=data.x.shape[1],
        hidden_dim=128,
        time_dim=32,
        num_layers=2,
        use_velocity=True
    )
    rel_auc, rel_val = train_gnn(rel_tgnn, data, tr, va, te, temporal=True, name="Relative TempGNN")
    results.append({'Model': 'Relative Temporal GNN', 'Test PR-AUC': rel_auc, 'Type': 'Relative'})

    # Summary
    print("="*70)
    print("  FINAL RESULTS")
    print("="*70)

    df = pd.DataFrame(results)
    baseline = df[df['Model'] == 'Logistic Regression']['Test PR-AUC'].values[0]
    df['Improvement vs LR'] = ((df['Test PR-AUC'] - baseline) / baseline * 100).round(1)

    print(df.to_string(index=False))

    # Highlight best
    best_model = df.loc[df['Test PR-AUC'].idxmax()]
    print(f"\n{'='*70}")
    print(f"  BEST MODEL: {best_model['Model']}")
    print(f"  Test PR-AUC: {best_model['Test PR-AUC']:.4f}")
    print(f"  Improvement: +{best_model['Improvement vs LR']:.1f}% over Logistic Regression")
    print("="*70)

    # Check if we hit target
    if rel_auc >= 0.60:
        print(f"\nSUCCESS! Relative Temporal GNN achieved {rel_auc:.1%} PR-AUC")
        print("   Target of 60-70% REACHED!")
    else:
        print(f"\nRelative Temporal GNN: {rel_auc:.1%} PR-AUC")
        print(f"   Gap to 60% target: {(0.60 - rel_auc)*100:.1f} percentage points")

    # Save
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/final_comparison.csv', index=False)
    print("\nResults saved to results/final_comparison.csv")

if __name__ == "__main__":
    main()
