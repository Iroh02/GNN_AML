"""
Feature-Only MLP Baseline on Elliptic (deconfounding the 37.6 pp claim)
=======================================================================

The LogisticRegression -> GraphSAGE comparison changes two things at once:
access to graph structure AND model class (linear vs deep non-linear with
focal loss). This script isolates the confound with a feature-only MLP that
matches the GNN's depth, width, loss, optimiser, schedule and seeds but never
sees an edge:

    MLP: Linear(165->128) -> ReLU -> Dropout -> Linear(128->128) -> BN -> ReLU
         -> classifier(128->64->2), focal loss, AdamW, cosine, 5 seeds.

If MLP ~ LogisticRegression, the 37.6 pp effect is graph structure.
If MLP >> LogisticRegression, part of the effect is model class and the
paper's claim must be attributed accordingly. Result is reported either way.

Output: results/mlp_baseline_elliptic.json / .csv
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
import warnings
warnings.filterwarnings('ignore')

from run_improved_gnn import get_masks, AdaptiveFocalLoss
from run_multiseed_elliptic import SEEDS, EPOCHS, PATIENCE, DEVICE, set_seed, get_data


class FeatureOnlyMLP(nn.Module):
    """GraphSAGE-matched capacity, zero graph access."""

    def __init__(self, in_dim, hid=128, drop=0.3):
        super().__init__()
        self.nenc = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, hid))
        self.mid = nn.Sequential(
            nn.Linear(hid, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(drop))
        self.clf = nn.Sequential(
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hid // 2, 2))

    def forward(self, x):
        return self.clf(self.mid(self.nenc(x)))


def run_seed(data, masks, seed):
    train_mask, val_mask, test_mask = masks
    set_seed(seed)
    model = FeatureOnlyMLP(data.x.shape[1]).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = AdaptiveFocalLoss(gamma=2.0, alpha=0.75)

    y_val = data.y[val_mask].cpu().numpy()
    best_val_pr, best_state, wait = 0.0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_probs = F.softmax(model(data.x)[val_mask], dim=1)[:, 1].cpu().numpy()
        val_pr = average_precision_score(y_val, val_probs)
        if val_pr > best_val_pr:
            best_val_pr, wait = val_pr, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x)
        probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
        preds = out[test_mask].argmax(dim=1).cpu().numpy()
    y = data.y[test_mask].cpu().numpy()
    return {
        'seed': seed,
        'test_pr_auc': float(average_precision_score(y, probs)),
        'test_roc_auc': float(roc_auc_score(y, probs)),
        'test_f1': float(f1_score(y, preds, zero_division=0)),
        'val_pr_auc': float(best_val_pr),
    }


def main():
    print("Feature-only MLP baseline on Elliptic "
          f"(seeds {SEEDS}, {EPOCHS} epochs, patience {PATIENCE})")
    data = get_data()
    masks = get_masks(data)
    data = data.to(DEVICE)
    masks = tuple(m.to(DEVICE) for m in masks)

    records = [run_seed(data, masks, s) for s in SEEDS]
    for r in records:
        print(f"  seed {r['seed']:>5}: PR {r['test_pr_auc']:.4f} | "
              f"ROC {r['test_roc_auc']:.4f} | F1 {r['test_f1']:.4f}")

    pr = np.array([r['test_pr_auc'] for r in records]) * 100
    roc = np.array([r['test_roc_auc'] for r in records]) * 100
    print(f"\n  MLP (no graph): PR {pr.mean():.2f}+/-{pr.std(ddof=1):.2f}  "
          f"ROC {roc.mean():.2f}+/-{roc.std(ddof=1):.2f}")

    os.makedirs('results', exist_ok=True)
    with open('results/mlp_baseline_elliptic.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'per_run': records}, f, indent=2)
    import pandas as pd
    pd.DataFrame(records).to_csv('results/mlp_baseline_elliptic.csv', index=False)
    print("Saved: results/mlp_baseline_elliptic.json/.csv")


if __name__ == "__main__":
    main()
