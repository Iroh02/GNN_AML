"""
Multi-Seed Resplit Diagnostic
=============================

Upgrades the M2 resplit diagnostic (paper Table V) from single-seed (42) to
five seeds, removing the last single-seed number in the camera-ready.

Protocol mirrors run_riskmagnn_resplit.py exactly, aligned with the unified
multi-seed protocol: RiskMAGNN Large (d=192, L=3, dropout=0.40) on the
stratified resplit (data/hbtbd_resplit/), temporal validation at ts>=31,
fixed 400-epoch budget, best-val checkpoint, seeds {0, 1, 7, 42, 123}.

Output: results/resplit_seeds.json / .csv
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from src.models.riskmagnn import create_riskmagnn
from run_multiseed_all import EPOCHS, NO_EARLY_STOP, DEVICE, set_seed
from run_riskmagnn import (
    load_hbtbd_data, extract_subgraph_adj, train_model, evaluate,
)

SEEDS = [0, 1, 7, 42, 123]


def prepare_resplit():
    train_path = 'data/hbtbd_resplit/train/'
    test_path = 'data/hbtbd_resplit/test/'

    print("[1] Loading resplit data...")
    train_features, train_labels, train_ts, train_adj, scaler = \
        load_hbtbd_data(train_path, normalize=True)
    test_features, test_labels, _, test_adj, _ = \
        load_hbtbd_data(test_path, normalize=True, scaler=scaler)

    if (train_labels == 1).float().mean() > 0.5:
        train_labels = 1 - train_labels
        test_labels = 1 - test_labels

    val_idx = torch.where(train_ts >= 31)[0]
    val_features = train_features[val_idx]
    val_labels = train_labels[val_idx]
    val_adj = [extract_subgraph_adj(a, val_idx) for a in train_adj]

    to = lambda x: x.to(DEVICE)
    print(f"  Train: {len(train_labels):,} nodes "
          f"({(train_labels == 1).float().mean()*100:.1f}% illicit) | "
          f"Val: {len(val_labels):,} | Test: {len(test_labels):,}")

    return {
        'train_features': to(train_features), 'train_labels': to(train_labels),
        'train_adj': [to(a) for a in train_adj],
        'val_features': to(val_features), 'val_labels': to(val_labels),
        'val_adj': [to(a) for a in val_adj],
        'test_features': to(test_features), 'test_labels': to(test_labels),
        'test_adj': [to(a) for a in test_adj],
    }


def main():
    print("=" * 70)
    print("  MULTI-SEED RESPLIT DIAGNOSTIC -- RiskMAGNN Large, "
          f"seeds {SEEDS}")
    print("=" * 70 + "\n")

    data = prepare_resplit()

    records = []
    for seed in SEEDS:
        print("=" * 70)
        print(f"  RiskMAGNN (Large, resplit) | seed {seed}")
        print("=" * 70)

        set_seed(seed)
        model = create_riskmagnn(
            num_features=data['train_features'].shape[1],
            hidden_dim=192, num_layers=3, num_metapaths=3, dropout=0.40,
        ).to(DEVICE)

        model, val_pr = train_model(
            model,
            data['train_features'], data['train_labels'], data['train_adj'],
            data['val_features'], data['val_labels'], data['val_adj'],
            epochs=EPOCHS, patience=NO_EARLY_STOP,
            name=f"RiskMAGNN Large resplit seed={seed}",
        )

        m = evaluate(model, data['test_features'], data['test_labels'],
                     data['test_adj'])
        gap = val_pr - m['pr_auc']
        print(f"  -> Val PR {val_pr:.4f} | Test PR {m['pr_auc']:.4f} | "
              f"gap {gap*100:.2f} pp | ROC {m['roc_auc']:.4f} | F1 {m['f1']:.4f}\n")

        records.append({
            'seed': seed,
            'val_pr_auc': float(val_pr),
            'test_pr_auc': float(m['pr_auc']),
            'gap_pp': float(gap * 100),
            'test_roc_auc': float(m['roc_auc']),
            'test_f1': float(m['f1']),
        })

    def stats(key):
        v = np.array([r[key] for r in records], dtype=float)
        return float(v.mean()), float(v.std(ddof=1))

    print("=" * 70)
    print("  SUMMARY (mean +/- std over seeds)")
    print("=" * 70)
    for key, label in [('val_pr_auc', 'Val PR-AUC'),
                       ('test_pr_auc', 'Test PR-AUC'),
                       ('gap_pp', 'Val-test gap (pp)')]:
        mu, sd = stats(key)
        scale = 1 if key == 'gap_pp' else 100
        print(f"  {label:<20} {mu*scale:6.2f} +/- {sd*scale:.2f}")

    os.makedirs('results', exist_ok=True)
    with open('results/resplit_seeds.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'epochs': EPOCHS, 'per_run': records}, f,
                  indent=2)
    import pandas as pd
    pd.DataFrame(records).to_csv('results/resplit_seeds.csv', index=False)
    print("\nSaved: results/resplit_seeds.json and results/resplit_seeds.csv")


if __name__ == "__main__":
    main()
