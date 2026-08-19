"""
Feature-Only MLP Baseline on HBTBD (10 seeds)
=============================================

Companion to run_mlp_baseline.py: tests whether the heterogeneous GNNs'
performance on HBTBD comes from metapath message passing or from the node
features alone (which already include 1-hop and 2-hop aggregates, indices
41-164). Same protocol as run_multiseed_extended.py -- same seeds, focal
loss, AdamW, warmup+cosine, 400-epoch budget, best-val checkpoint -- but the
model never sees a metapath adjacency.

Output: results/mlp_baseline_hbtbd.json / .csv
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

from run_multiseed_all import EPOCHS, DEVICE, set_seed, prepare_data
from run_riskmagnn import FocalLoss

SEEDS = [0, 1, 7, 42, 123, 11, 21, 77, 2024, 31337]


class FeatureOnlyMLP(nn.Module):
    """Capacity-matched to RiskMAGNN Base's non-graph components."""

    def __init__(self, in_dim, hid=128, drop=0.35):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.body = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(drop),
        )
        self.clf = nn.Sequential(
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(drop / 2),
            nn.Linear(hid // 2, 2))

    def forward(self, x):
        return self.clf(self.body(self.input_norm(x)))


def run_seed(data, seed):
    set_seed(seed)
    model = FeatureOnlyMLP(data['train_features'].shape[1]).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    warmup = 15

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        prog = (epoch - warmup) / (EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75, smoothing=0.1)

    y_val = data['val_labels'].cpu().numpy()
    best_val_pr, best_state = 0.0, None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(data['train_features']), data['train_labels'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vp = F.softmax(model(data['val_features']), dim=1)[:, 1].cpu().numpy()
            val_pr = average_precision_score(y_val, vp)
            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data['test_features'])
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
    y = data['test_labels'].cpu().numpy()
    return {
        'seed': seed,
        'val_pr_auc': float(best_val_pr),
        'test_pr_auc': float(average_precision_score(y, probs)),
        'test_roc_auc': float(roc_auc_score(y, probs)),
        'test_f1': float(f1_score(y, preds, zero_division=0)),
    }


def main():
    print(f"Feature-only MLP on HBTBD (seeds {SEEDS}, {EPOCHS} epochs)")
    data = prepare_data()
    records = [run_seed(data, s) for s in SEEDS]
    for r in records:
        print(f"  seed {r['seed']:>5}: PR {r['test_pr_auc']:.4f} | "
              f"ROC {r['test_roc_auc']:.4f} | F1 {r['test_f1']:.4f}")
    pr = np.array([r['test_pr_auc'] for r in records]) * 100
    roc = np.array([r['test_roc_auc'] for r in records]) * 100
    f1 = np.array([r['test_f1'] for r in records]) * 100
    print(f"\n  MLP (no metapaths): PR {pr.mean():.2f}+/-{pr.std(ddof=1):.2f}  "
          f"ROC {roc.mean():.2f}+/-{roc.std(ddof=1):.2f}  "
          f"F1 {f1.mean():.2f}+/-{f1.std(ddof=1):.2f}")

    os.makedirs('results', exist_ok=True)
    with open('results/mlp_baseline_hbtbd.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'per_run': records}, f, indent=2)
    import pandas as pd
    pd.DataFrame(records).to_csv('results/mlp_baseline_hbtbd.csv', index=False)
    print("Saved: results/mlp_baseline_hbtbd.json/.csv")


if __name__ == "__main__":
    main()
