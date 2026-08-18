"""
Multi-Seed Evaluation on the Elliptic Chronological Split
=========================================================

Removes the single-seed caveat from the Elliptic tables (paper Tables I and II)
by re-running every homogeneous model across 5 seeds under one protocol.

Models / variants
-----------------
  LogisticRegression      feature-only baseline, class-weighted, no graph
  StaticGraphSAGE         2-layer SAGEConv, no temporal input
  TemporalGNN             GraphSAGE + sinusoidal timestep encoding + skip
  TemporalGNN (no skip)   the *controlled* temporal baseline
  No temporal (no skip)   controlled baseline with the timestep input removed

The last two form the controlled pair behind the paper's temporal claim: they
differ ONLY in whether the timestep position encoding is fed to the network,
with skip connections and LayerNorm removed from both so the comparison is not
confounded by architectural stabilizers. Because both variants see the same
seeds, the significance analysis pairs them by seed.

IMPORTANT SCOPE NOTE: the "no temporal" variant ablates *timestep position
encoding*, not temporal graph learning in general. Dedicated temporal graph
architectures (TGN, TGAT) are NOT evaluated here and no claim about them is
supported by this experiment.

Split: standard chronological -- train ts 1-29, val ts 30-39, test ts 40-49.

Outputs: results/multiseed_elliptic.json and results/multiseed_elliptic.csv
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import SAGEConv, BatchNorm
import warnings
warnings.filterwarnings('ignore')

from run_improved_gnn import load_data, get_masks, AdaptiveFocalLoss

SEEDS = [0, 1, 7, 42, 123]
EPOCHS = 300
PATIENCE = 50
CACHE = 'data/processed/elliptic_multiseed_cache.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ConfigurableTemporalGNN(nn.Module):
    """
    GraphSAGE encoder with two independently switchable components.

    Args:
        use_temporal: feed sinusoidal timestep position encoding alongside
            node features. This is the component the temporal claim is about.
        use_skip: residual/skip connections from the input projection into the
            first conv layer and between subsequent layers.
    """

    def __init__(self, in_dim, hid=128, tdim=32, layers=2, drop=0.3,
                 use_temporal=True, use_skip=True):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_skip = use_skip
        self.drop = drop

        self.nenc = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, hid))

        if use_temporal:
            self.tenc = nn.Linear(1, tdim)
            enc_dim = hid + tdim
        else:
            enc_dim = hid

        self.convs = nn.ModuleList([
            SAGEConv(enc_dim if i == 0 else hid, hid) for i in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hid) for _ in range(layers)])

        if use_skip:
            self.input_proj = nn.Linear(enc_dim, hid)

        self.clf = nn.Sequential(
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hid // 2, 2))

    def forward(self, x, ei, ts):
        h = self.nenc(x)
        if self.use_temporal:
            t = torch.sin(self.tenc(ts.float().unsqueeze(-1) / 49.0))
            h = torch.cat([h, t], -1)

        h_skip = self.input_proj(h) if self.use_skip else None

        for i, (c, b) in enumerate(zip(self.convs, self.bns)):
            h_in = h
            h = c(h, ei)
            h = b(h)
            h = F.relu(h)
            h = F.dropout(h, self.drop, self.training)
            if self.use_skip:
                h = h + (h_skip if i == 0 else h_in)
        return self.clf(h)


def metrics_from(probs, preds, targets):
    return {
        'test_pr_auc': float(average_precision_score(targets, probs)),
        'test_roc_auc': float(roc_auc_score(targets, probs)),
        'test_f1': float(f1_score(targets, preds, zero_division=0)),
    }


def run_logreg(data, masks, seed):
    """Feature-only baseline. No graph structure."""
    train_mask, _, test_mask = masks
    set_seed(seed)

    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    tr, te = train_mask.cpu().numpy(), test_mask.cpu().numpy()

    clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                             random_state=seed)
    clf.fit(X[tr], y[tr])
    probs = clf.predict_proba(X[te])[:, 1]
    preds = clf.predict(X[te])
    return metrics_from(probs, preds, y[te])


def run_gnn(data, masks, seed, use_temporal, use_skip, name):
    train_mask, val_mask, test_mask = masks
    set_seed(seed)

    model = ConfigurableTemporalGNN(
        in_dim=data.x.shape[1], hid=128, tdim=32, layers=2, drop=0.3,
        use_temporal=use_temporal, use_skip=use_skip).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = AdaptiveFocalLoss(gamma=2.0, alpha=0.75)

    y_val = data.y[val_mask].cpu().numpy()
    best_val_pr, best_state, wait = 0.0, None, 0

    print(f"\n  Training {name} (seed {seed})... "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    for epoch in range(EPOCHS):
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
            val_probs = F.softmax(out[val_mask], dim=1)[:, 1].cpu().numpy()
            val_pr = average_precision_score(y_val, val_probs)

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch + 1:3d}: Loss={loss.item():.4f}, "
                  f"Val PR-AUC={val_pr:.4f}, Best={best_val_pr:.4f}")

        if wait >= PATIENCE:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.timestep)
        probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
        preds = out[test_mask].argmax(dim=1).cpu().numpy()

    m = metrics_from(probs, preds, data.y[test_mask].cpu().numpy())
    m['val_pr_auc'] = float(best_val_pr)
    return m


# name -> callable(data, masks, seed) -> metrics dict
VARIANTS = [
    ('LogisticRegression',
     lambda d, m, s: run_logreg(d, m, s)),
    ('StaticGraphSAGE',
     lambda d, m, s: run_gnn(d, m, s, False, True, 'StaticGraphSAGE')),
    ('TemporalGNN',
     lambda d, m, s: run_gnn(d, m, s, True, True, 'TemporalGNN')),
    ('TemporalGNN (no skip)',
     lambda d, m, s: run_gnn(d, m, s, True, False, 'TemporalGNN (no skip)')),
    ('No temporal (no skip)',
     lambda d, m, s: run_gnn(d, m, s, False, False, 'No temporal (no skip)')),
]


def get_data():
    """Load Elliptic, caching the parsed tensors (the raw CSV is ~690 MB)."""
    if os.path.exists(CACHE):
        print(f"Loading cached Elliptic tensors from {CACHE}...")
        return torch.load(CACHE, weights_only=False)
    data = load_data()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(data, CACHE)
    return data


def main():
    print("=" * 70)
    print("  MULTI-SEED EVALUATION -- Elliptic Chronological Split")
    print(f"  Seeds: {SEEDS} | {EPOCHS} epochs, patience {PATIENCE}")
    print(f"  Device: {DEVICE}")
    print("=" * 70 + "\n")

    data = get_data()
    masks = get_masks(data)
    train_mask, val_mask, test_mask = masks

    print(f"\n  Train: {train_mask.sum().item():,} nodes "
          f"({data.y[train_mask].float().mean()*100:.2f}% illicit)")
    print(f"  Val:   {val_mask.sum().item():,} nodes "
          f"({data.y[val_mask].float().mean()*100:.2f}% illicit)")
    print(f"  Test:  {test_mask.sum().item():,} nodes "
          f"({data.y[test_mask].float().mean()*100:.2f}% illicit)\n")

    data = data.to(DEVICE)
    masks = tuple(m.to(DEVICE) for m in masks)

    records = []
    for name, fn in VARIANTS:
        print("=" * 70)
        print(f"  {name}")
        print("=" * 70)
        for seed in SEEDS:
            m = fn(data, masks, seed)
            m.update({'variant': name, 'seed': seed})
            records.append(m)
            print(f"  -> seed {seed}: PR {m['test_pr_auc']:.4f} | "
                  f"ROC {m['test_roc_auc']:.4f} | F1 {m['test_f1']:.4f}")
        print()

    def stats(vals):
        v = np.asarray(vals, dtype=float)
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    summary = {}
    for name, _ in VARIANTS:
        rs = [r for r in records if r['variant'] == name]
        summary[name] = {}
        for key in ['test_pr_auc', 'test_roc_auc', 'test_f1']:
            mean, std = stats([r[key] for r in rs])
            summary[name][key] = {'mean': mean, 'std': std}

    print("=" * 70)
    print("  PER-SEED TEST PR-AUC")
    print("=" * 70)
    header = "  {:<24}".format("Variant") + "".join(f"{s:>8}" for s in SEEDS) + f"{'mean':>9}{'std':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, _ in VARIANTS:
        row = f"  {name:<24}"
        for s in SEEDS:
            v = next(r['test_pr_auc'] for r in records
                     if r['variant'] == name and r['seed'] == s)
            row += f"{v*100:>7.2f} "
        mu = summary[name]['test_pr_auc']['mean'] * 100
        sd = summary[name]['test_pr_auc']['std'] * 100
        row += f"{mu:>8.2f}{sd:>7.2f}"
        print(row)

    print("\n" + "=" * 70)
    print("  SUMMARY (mean +/- sample std across seeds)")
    print("=" * 70)
    for name, _ in VARIANTS:
        pr = summary[name]['test_pr_auc']
        roc = summary[name]['test_roc_auc']
        f1 = summary[name]['test_f1']
        print(f"  {name:<24} PR-AUC {pr['mean']*100:5.2f}+/-{pr['std']*100:.2f}  "
              f"ROC {roc['mean']*100:5.2f}+/-{roc['std']*100:.2f}  "
              f"F1 {f1['mean']*100:5.2f}+/-{f1['std']*100:.2f}")

    os.makedirs('results', exist_ok=True)
    with open('results/multiseed_elliptic.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'epochs': EPOCHS,
                   'per_run': records, 'summary': summary}, f, indent=2)
    import pandas as pd
    pd.DataFrame(records).to_csv('results/multiseed_elliptic.csv', index=False)
    print("\nSaved: results/multiseed_elliptic.json and results/multiseed_elliptic.csv")


if __name__ == "__main__":
    main()
