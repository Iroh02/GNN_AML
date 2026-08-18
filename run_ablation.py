"""
Ablation Studies for Temporal GNN
=================================

Prove the value of each component by removing them one at a time:
1. Full Temporal GNN (baseline)
2. Without temporal encoding (static)
3. Without skip connections
4. Without feature standardization
5. Smaller model (fewer layers/dims)

This validates the proposal's claim that temporal modeling improves AML detection.
"""

import os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
import warnings
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data(standardize=True):
    df = pd.read_csv('data/raw/elliptic_txs_features.csv', header=None)
    node_ids = df.iloc[:, 0].values
    timesteps = df.iloc[:, 1].values.astype(np.int64)
    features = df.iloc[:, 2:].values.astype(np.float32)
    if standardize:
        features = StandardScaler().fit_transform(features)
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

# Model variants for ablation
class FullTemporalGNN(nn.Module):
    """Full model with temporal encoding + skip connections"""
    def __init__(self, in_dim, hid=256, tdim=64, layers=3, drop=0.3):
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
            h_in = h
            h = F.dropout(F.relu(b(c(torch.cat([h, t], -1), ei))), self.drop, self.training) + h_in  # Skip connection
        return self.clf(h)

class NoTemporalGNN(nn.Module):
    """Without temporal encoding (static GNN)"""
    def __init__(self, in_dim, hid=256, layers=3, drop=0.3):
        super().__init__()
        self.nenc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.convs = nn.ModuleList([SAGEConv(hid, hid) for _ in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hid) for _ in range(layers)])
        self.clf = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid//2, 2))
        self.drop = drop
    def forward(self, x, ei, ts=None):
        h = self.nenc(x)
        for c, b in zip(self.convs, self.bns):
            h_in = h
            h = F.dropout(F.relu(b(c(h, ei))), self.drop, self.training) + h_in
        return self.clf(h)

class NoSkipGNN(nn.Module):
    """Without skip connections"""
    def __init__(self, in_dim, hid=256, tdim=64, layers=3, drop=0.3):
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
            h = F.dropout(F.relu(b(c(torch.cat([h, t], -1), ei))), self.drop, self.training)  # No skip
        return self.clf(h)

class SmallTemporalGNN(nn.Module):
    """Smaller model (128 hidden, 2 layers)"""
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
            h_in = h
            h = F.dropout(F.relu(b(c(torch.cat([h, t], -1), ei))), self.drop, self.training) + h_in
        return self.clf(h)

def train_model(model, data, tr, va, te, use_temporal=True, epochs=200, lr=0.001):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
    loss_fn = FocalLoss()
    best, state, wait = 0, None, 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.timestep) if use_temporal else model(data.x, data.edge_index)
        loss_fn(out[tr], data.y[tr]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.timestep) if use_temporal else model(data.x, data.edge_index)
            auc = average_precision_score(data.y[va].numpy(), F.softmax(out[va], 1)[:, 1].numpy())
        if auc > best:
            best, state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= 25:
            break

    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.timestep) if use_temporal else model(data.x, data.edge_index)
        test_auc = average_precision_score(data.y[te].numpy(), F.softmax(out[te], 1)[:, 1].numpy())
    return test_auc, best

def main():
    print("="*60)
    print("  ABLATION STUDIES - TEMPORAL GNN COMPONENTS")
    print("="*60)

    # Load data with standardization
    print("\nLoading data...")
    data = load_data(standardize=True)
    tr, va, te = get_masks(data)

    results = []

    # 1. Full Temporal GNN (baseline)
    print("\n[1/5] Full Temporal GNN (baseline)...")
    model = FullTemporalGNN(data.x.shape[1], hid=256, tdim=64, layers=3)
    test_auc, val_auc = train_model(model, data, tr, va, te, use_temporal=True)
    results.append({'Variant': 'Full Temporal GNN', 'Test PR-AUC': test_auc, 'Val PR-AUC': val_auc})
    print(f"    Test PR-AUC: {test_auc:.4f}")

    # 2. Without temporal encoding
    print("\n[2/5] Without Temporal Encoding (Static GNN)...")
    model = NoTemporalGNN(data.x.shape[1], hid=256, layers=3)
    test_auc, val_auc = train_model(model, data, tr, va, te, use_temporal=False)
    results.append({'Variant': 'No Temporal Encoding', 'Test PR-AUC': test_auc, 'Val PR-AUC': val_auc})
    print(f"    Test PR-AUC: {test_auc:.4f}")

    # 3. Without skip connections
    print("\n[3/5] Without Skip Connections...")
    model = NoSkipGNN(data.x.shape[1], hid=256, tdim=64, layers=3)
    test_auc, val_auc = train_model(model, data, tr, va, te, use_temporal=True)
    results.append({'Variant': 'No Skip Connections', 'Test PR-AUC': test_auc, 'Val PR-AUC': val_auc})
    print(f"    Test PR-AUC: {test_auc:.4f}")

    # 4. Smaller model
    print("\n[4/5] Smaller Model (128 hidden, 2 layers)...")
    model = SmallTemporalGNN(data.x.shape[1], hid=128, tdim=32, layers=2)
    test_auc, val_auc = train_model(model, data, tr, va, te, use_temporal=True)
    results.append({'Variant': 'Smaller Model', 'Test PR-AUC': test_auc, 'Val PR-AUC': val_auc})
    print(f"    Test PR-AUC: {test_auc:.4f}")

    # 5. Without feature standardization
    print("\n[5/5] Without Feature Standardization...")
    data_no_std = load_data(standardize=False)
    model = FullTemporalGNN(data_no_std.x.shape[1], hid=256, tdim=64, layers=3)
    test_auc, val_auc = train_model(model, data_no_std, tr, va, te, use_temporal=True)
    results.append({'Variant': 'No Standardization', 'Test PR-AUC': test_auc, 'Val PR-AUC': val_auc})
    print(f"    Test PR-AUC: {test_auc:.4f}")

    # Summary
    print("\n" + "="*60)
    print("  ABLATION RESULTS SUMMARY")
    print("="*60)

    df = pd.DataFrame(results)
    baseline = df[df['Variant'] == 'Full Temporal GNN']['Test PR-AUC'].values[0]
    df['Delta'] = df['Test PR-AUC'] - baseline
    df['% Change'] = (df['Delta'] / baseline * 100).round(1)

    print(df.to_string(index=False))

    print("\n" + "-"*60)
    print("KEY FINDINGS:")
    print("-"*60)

    no_temporal = df[df['Variant'] == 'No Temporal Encoding']['Test PR-AUC'].values[0]
    temporal_gain = baseline - no_temporal
    print(f"  - Temporal encoding adds: +{temporal_gain:.4f} PR-AUC ({temporal_gain/no_temporal*100:.1f}% improvement)")

    no_skip = df[df['Variant'] == 'No Skip Connections']['Test PR-AUC'].values[0]
    skip_gain = baseline - no_skip
    print(f"  - Skip connections add: +{skip_gain:.4f} PR-AUC")

    no_std = df[df['Variant'] == 'No Standardization']['Test PR-AUC'].values[0]
    std_gain = baseline - no_std
    print(f"  - Feature standardization adds: +{std_gain:.4f} PR-AUC")

    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/ablation_results.csv', index=False)
    print("\nResults saved to results/ablation_results.csv")

if __name__ == "__main__":
    main()
