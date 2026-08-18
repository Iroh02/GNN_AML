"""
Unified Multi-Seed Evaluation on the Official HBTBD Split
=========================================================

Trains THREE models under one identical, clean protocol so the comparison
between them is finally apples-to-apples:

  * SimpleHeteroGNN   (hidden=128, L=2, dropout=0.30)
  * RiskMAGNN (Base)  (hidden=128, L=2, dropout=0.35)
  * RiskMAGNN (Large) (hidden=192, L=3, dropout=0.40)

Shared protocol (the clean one from run_riskmagnn.py), applied to all models:
  * StandardScaler-normalized features (fit on train only)
  * Strict temporal validation split: train ts 1-34, validate on ts 31-34
    using REAL metapath adjacencies (no identity-matrix shortcut)
  * Focal loss, AdamW, warmup + cosine schedule
  * Fixed 400-epoch budget, best-validation-checkpoint selection
    (early stopping DISABLED -- it was fragile and collapsed some seeds)
  * Seeds: 0, 1, 7, 42, 123, each freshly set before model construction

Why this differs from the published tables:
  * The original SimpleHeteroGNN run (run_hbtbd.py) used a random val split
    with IDENTITY metapath adjacencies and validation-in-training leakage,
    so its 60.4% / 98.7%-val numbers were not comparable to RiskMAGNN.
  * The original RiskMAGNN 69.02% was a single-seed artifact of training
    Base then Large in one process under one global seed.

Outputs: results/multiseed_all.json and results/multiseed_all.csv
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from src.models.riskmagnn import create_riskmagnn
from src.models.magnn import create_simple_hetero_gnn
from run_riskmagnn import (
    load_hbtbd_data,
    extract_subgraph_adj,
    train_model,
    evaluate,
)

SEEDS = [0, 1, 7, 42, 123]
EPOCHS = 400
NO_EARLY_STOP = 10 ** 9  # patience high enough to disable early stopping
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Each model: a label and a builder(num_features) -> nn.Module
MODEL_CONFIGS = [
    {
        'name': 'SimpleHeteroGNN',
        'build': lambda nf: create_simple_hetero_gnn(
            num_features=nf, hidden_dim=128, num_layers=2,
            num_metapaths=3, dropout=0.30),
    },
    {
        'name': 'RiskMAGNN (Base)',
        'build': lambda nf: create_riskmagnn(
            num_features=nf, hidden_dim=128, num_layers=2,
            num_metapaths=3, dropout=0.35),
    },
    {
        'name': 'RiskMAGNN (Large)',
        'build': lambda nf: create_riskmagnn(
            num_features=nf, hidden_dim=192, num_layers=3,
            num_metapaths=3, dropout=0.40),
    },
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def to_device(x):
    return x.to(DEVICE)


def prepare_data():
    """Load official HBTBD split + clean temporal val split (no RNG -> once)."""
    train_path = 'data/hbtbd/HBTBD/data/train/'
    test_path = 'data/hbtbd/HBTBD/data/test/'

    print("[1] Loading data...")
    train_features, train_labels, train_ts, train_adj, scaler = \
        load_hbtbd_data(train_path, normalize=True)
    test_features, test_labels, test_ts, test_adj, _ = \
        load_hbtbd_data(test_path, normalize=True, scaler=scaler)

    if (train_labels == 1).float().mean() > 0.5:
        train_labels = 1 - train_labels
        test_labels = 1 - test_labels

    # Strict temporal validation split: train ts 1-30, val ts 31-34
    val_idx = torch.where(train_ts >= 31)[0]
    val_features = train_features[val_idx]
    val_labels = train_labels[val_idx]
    val_adj = [extract_subgraph_adj(a, val_idx) for a in train_adj]

    print(f"  Train (full ts 1-34): {len(train_labels):,} nodes "
          f"({(train_labels == 1).float().mean()*100:.1f}% illicit)")
    print(f"  Val   (ts 31-34):     {len(val_labels):,} nodes")
    print(f"  Test:                 {len(test_labels):,} nodes "
          f"({(test_labels == 1).float().mean()*100:.1f}% illicit)")
    print(f"  Device: {DEVICE}\n")

    return {
        'train_features': to_device(train_features),
        'train_labels': to_device(train_labels),
        'train_adj': [to_device(a) for a in train_adj],
        'val_features': to_device(val_features),
        'val_labels': to_device(val_labels),
        'val_adj': [to_device(a) for a in val_adj],
        'test_features': to_device(test_features),
        'test_labels': to_device(test_labels),
        'test_adj': [to_device(a) for a in test_adj],
    }


def run_one(cfg: dict, seed: int, data: dict) -> dict:
    print("=" * 70)
    print(f"  {cfg['name']} | seed {seed}")
    print("=" * 70)

    set_seed(seed)
    model = cfg['build'](data['train_features'].shape[1]).to(DEVICE)

    model, val_pr = train_model(
        model,
        data['train_features'], data['train_labels'], data['train_adj'],
        data['val_features'], data['val_labels'], data['val_adj'],
        epochs=EPOCHS, patience=NO_EARLY_STOP,
        name=f"{cfg['name']} seed={seed}",
    )

    m = evaluate(model, data['test_features'], data['test_labels'], data['test_adj'])
    print(f"  -> Val PR {val_pr:.4f} | Test PR {m['pr_auc']:.4f} | "
          f"ROC {m['roc_auc']:.4f} | F1 {m['f1']:.4f}\n")

    return {
        'model': cfg['name'], 'seed': seed,
        'val_pr_auc': float(val_pr),
        'test_pr_auc': float(m['pr_auc']),
        'test_roc_auc': float(m['roc_auc']),
        'test_f1': float(m['f1']),
        'test_precision': float(m['precision']),
        'test_recall': float(m['recall']),
    }


def summarize(records: list) -> dict:
    def stats(vals):
        vals = np.asarray(vals, dtype=float)
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        return float(vals.mean()), std

    out = {}
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        rs = [r for r in records if r['model'] == name]
        out[name] = {}
        for key in ['test_pr_auc', 'test_roc_auc', 'test_f1']:
            mean, std = stats([r[key] for r in rs])
            out[name][key] = {'mean': mean, 'std': std}
    return out


def main():
    print("=" * 70)
    print("  UNIFIED MULTI-SEED EVALUATION -- Official HBTBD Split")
    print(f"  Models: {[c['name'] for c in MODEL_CONFIGS]}")
    print(f"  Seeds: {SEEDS} | Fixed {EPOCHS}-epoch budget, best-val checkpoint")
    print("=" * 70 + "\n")

    data = prepare_data()

    records = []
    for cfg in MODEL_CONFIGS:
        for seed in SEEDS:
            records.append(run_one(cfg, seed, data))

    summary = summarize(records)

    # Per-seed table
    print("\n" + "=" * 70)
    print("  PER-SEED TEST PR-AUC")
    print("=" * 70)
    header = "  {:<20}".format("Model") + "".join(f"{s:>8}" for s in SEEDS) + f"{'mean':>9}{'std':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        row = f"  {name:<20}"
        for s in SEEDS:
            v = next(r['test_pr_auc'] for r in records if r['model'] == name and r['seed'] == s)
            row += f"{v*100:>7.2f} "
        mu = summary[name]['test_pr_auc']['mean'] * 100
        sd = summary[name]['test_pr_auc']['std'] * 100
        row += f"{mu:>8.2f}{sd:>7.2f}"
        print(row)

    # Summary + gaps
    print("\n" + "=" * 70)
    print("  SUMMARY (mean +/- sample std across seeds)")
    print("=" * 70)
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        pr = summary[name]['test_pr_auc']
        roc = summary[name]['test_roc_auc']
        f1 = summary[name]['test_f1']
        print(f"  {name:<20} PR-AUC {pr['mean']*100:5.2f}+/-{pr['std']*100:.2f}  "
              f"ROC {roc['mean']*100:5.2f}+/-{roc['std']*100:.2f}  "
              f"F1 {f1['mean']*100:5.2f}+/-{f1['std']*100:.2f}")

    large = summary['RiskMAGNN (Large)']['test_pr_auc']
    base = summary['RiskMAGNN (Base)']['test_pr_auc']
    simple = summary['SimpleHeteroGNN']['test_pr_auc']

    def gap(a, b):
        # difference of means and pooled std (independent runs)
        d = (a['mean'] - b['mean']) * 100
        pooled = ((a['std'] ** 2 + b['std'] ** 2) ** 0.5) * 100
        return d, pooled

    print("\n  Gaps (mean difference +/- combined std):")
    d1, s1 = gap(large, base)
    d2, s2 = gap(large, simple)
    print(f"    Large - Base          : {d1:+.2f} +/- {s1:.2f} pp")
    print(f"    Large - SimpleHeteroGNN: {d2:+.2f} +/- {s2:.2f} pp")

    os.makedirs('results', exist_ok=True)
    with open('results/multiseed_all.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'epochs': EPOCHS,
                   'per_run': records, 'summary': summary}, f, indent=2)
    import pandas as pd
    pd.DataFrame(records).to_csv('results/multiseed_all.csv', index=False)
    print("\nSaved: results/multiseed_all.json and results/multiseed_all.csv")


if __name__ == "__main__":
    main()
