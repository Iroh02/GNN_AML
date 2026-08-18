"""
Test with random split (not temporal) to compare with literature values
"""
import os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
import warnings
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data():
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

    y = torch.tensor(labels, dtype=torch.long)
    print(f"  Illicit: {(y==1).sum().item()}, Licit: {(y==0).sum().item()}")
    return Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edge_index,
                y=y, timestep=torch.tensor(timesteps, dtype=torch.long))

def get_random_masks(data, seed=42):
    """Random stratified split instead of temporal"""
    labeled_idx = torch.where(data.y != -1)[0].numpy()
    labels = data.y[labeled_idx].numpy()

    train_idx, temp_idx = train_test_split(labeled_idx, test_size=0.4, stratify=labels, random_state=seed)
    temp_labels = data.y[temp_idx].numpy()
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"  Train: {train_mask.sum().item()} (Illicit: {((data.y==1) & train_mask).sum().item()})")
    print(f"  Val: {val_mask.sum().item()} (Illicit: {((data.y==1) & val_mask).sum().item()})")
    print(f"  Test: {test_mask.sum().item()} (Illicit: {((data.y==1) & test_mask).sum().item()})")
    return train_mask, val_mask, test_mask

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hid=256, layers=2, drop=0.5):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim if i==0 else hid, hid) for i in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hid) for _ in range(layers)])
        self.clf = nn.Linear(hid, 2)
        self.drop = drop
    def forward(self, x, ei):
        for c, b in zip(self.convs, self.bns):
            x = F.dropout(F.relu(b(c(x, ei))), self.drop, self.training)
        return self.clf(x)

def train_gnn(model, data, tr, va, te, epochs=200, lr=0.01):
    y_tr = data.y[tr]
    w = torch.tensor([1.0, (y_tr==0).sum().float()/(y_tr==1).sum().float()])
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best, state, wait = 0, None, 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        criterion(out[tr], data.y[tr]).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            auc = average_precision_score(data.y[va].numpy(), F.softmax(out[va], 1)[:, 1].numpy())
        if auc > best: best, state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: wait += 1
        if (ep+1) % 50 == 0: print(f"    Ep {ep+1}: Val={auc:.4f} Best={best:.4f}")
        if wait >= 20: break

    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        return average_precision_score(data.y[te].numpy(), F.softmax(out[te], 1)[:, 1].numpy())

def main():
    print("="*55 + "\n  RANDOM SPLIT EXPERIMENT (like literature)\n" + "="*55)
    data = load_data()
    tr, va, te = get_random_masks(data)
    results = {}

    # Logistic Regression
    print("\n[1/2] Logistic Regression...")
    X, y = data.x.numpy(), data.y.numpy()
    w = (y[tr.numpy()] == 0).sum() / (y[tr.numpy()] == 1).sum()
    lr = LogisticRegression(class_weight={0:1, 1:w}, C=1.0, max_iter=1000, random_state=SEED)
    lr.fit(X[tr.numpy()], y[tr.numpy()])
    results['Logistic Regression'] = average_precision_score(y[te.numpy()], lr.predict_proba(X[te.numpy()])[:, 1])
    print(f"    PR-AUC: {results['Logistic Regression']:.4f}")

    # GraphSAGE
    print("\n[2/2] GraphSAGE...")
    gs = GraphSAGE(data.x.shape[1], hid=128, layers=2)
    results['GraphSAGE'] = train_gnn(gs, data, tr, va, te)
    print(f"    Test PR-AUC: {results['GraphSAGE']:.4f}")

    print("\n" + "="*55 + "\n  RANDOM SPLIT RESULTS\n" + "="*55)
    for k, v in results.items(): print(f"  {k}: PR-AUC = {v:.4f}")

    pd.DataFrame([{'Model': k, 'PR-AUC': v} for k, v in results.items()]).to_csv('results/summary_random.csv', index=False)

if __name__ == "__main__":
    main()
