"""
Hyperparameter Tuning Script for AML Detection
Key optimizations: Feature standardization, Focal Loss, larger models, skip connections
"""

import os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
import warnings
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data():
    print("Loading data with feature standardization...")
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

    y = torch.tensor(labels, dtype=torch.long)
    print(f"  Illicit: {(y==1).sum().item()}, Licit: {(y==0).sum().item()}")
    return Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edge_index,
                y=y, timestep=torch.tensor(timesteps, dtype=torch.long))

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

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hid=256, layers=3, drop=0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim if i==0 else hid, hid) for i in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hid) for _ in range(layers)])
        self.clf = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid//2, 2))
        self.drop = drop
    def forward(self, x, ei):
        for c, b in zip(self.convs, self.bns):
            x = F.dropout(F.relu(b(c(x, ei))), self.drop, self.training)
        return self.clf(x)

class TemporalGNN(nn.Module):
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
            h = F.dropout(F.relu(b(c(torch.cat([h, t], -1), ei))), self.drop, self.training) + h
        return self.clf(h)

def train_gnn(model, data, tr, va, te, temporal=False, epochs=300, lr=0.001):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
    loss_fn = FocalLoss()
    best, state, wait = 0, None, 0

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
        if auc > best: best, state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: wait += 1
        if (ep+1) % 50 == 0: print(f"    Ep {ep+1}: Val={auc:.4f} Best={best:.4f}")
        if wait >= 30: break

    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.timestep) if temporal else model(data.x, data.edge_index)
        return average_precision_score(data.y[te].numpy(), F.softmax(out[te], 1)[:, 1].numpy())

def main():
    print("="*55 + "\n  HYPERPARAMETER TUNING FOR AML DETECTION\n" + "="*55)
    data = load_data()
    tr, va, te = get_masks(data)
    results = {}

    # Logistic Regression
    print("\n[1/3] Logistic Regression...")
    X, y = data.x.numpy(), data.y.numpy()
    w = (y[tr.numpy()] == 0).sum() / (y[tr.numpy()] == 1).sum()
    lr = LogisticRegression(class_weight={0:1, 1:w}, C=0.1, max_iter=2000, random_state=SEED)
    lr.fit(X[tr.numpy()], y[tr.numpy()])
    results['Logistic Regression'] = average_precision_score(y[te.numpy()], lr.predict_proba(X[te.numpy()])[:, 1])
    print(f"    PR-AUC: {results['Logistic Regression']:.4f}")

    # GraphSAGE
    print("\n[2/3] GraphSAGE (256 hidden, 3 layers)...")
    gs = GraphSAGE(data.x.shape[1], hid=256, layers=3)
    results['GraphSAGE'] = train_gnn(gs, data, tr, va, te, temporal=False)
    print(f"    Test PR-AUC: {results['GraphSAGE']:.4f}")

    # Temporal GNN
    print("\n[3/3] Temporal GNN (256 hidden, 64 time dim, 3 layers)...")
    tgnn = TemporalGNN(data.x.shape[1], hid=256, tdim=64, layers=3)
    results['Temporal GNN'] = train_gnn(tgnn, data, tr, va, te, temporal=True)
    print(f"    Test PR-AUC: {results['Temporal GNN']:.4f}")

    print("\n" + "="*55 + "\n  TUNED RESULTS\n" + "="*55)
    for k, v in results.items(): print(f"  {k}: PR-AUC = {v:.4f}")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame([{'Model': k, 'PR-AUC': v} for k, v in results.items()]).to_csv('results/summary_tuned.csv', index=False)
    print("\nSaved to results/summary_tuned.csv")

if __name__ == "__main__":
    main()
