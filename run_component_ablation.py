"""
Multi-Seed Component Ablation for RiskMAGNN
===========================================

Isolates the contribution of the two components the paper claims as novel:

  * TransE-style metapath encoding  (use_transe)
  * Risk-biased inter-metapath attention (use_risk_bias)

Four variants in a 2x2 design, all built on the RiskMAGNN (Base) skeleton
(d=128, L=2, dropout=0.35) so that residual connections, LayerNorm, the
classifier head and the training schedule are held identical. The only thing
that changes is which of the two components is switched on:

  Full            : TransE + risk-biased attention   (the paper's model)
  -TransE         : mean aggregation + risk-biased attention
  -RiskBias       : TransE + uniform (unbiased) attention
  -Both           : mean aggregation + uniform attention (control)

Protocol is byte-for-byte the one used in run_multiseed_all.py, so these
numbers are directly comparable to the SimpleHeteroGNN / RiskMAGNN table:
  * StandardScaler features (fit on train only)
  * Strict temporal validation split: train ts 1-34, validate on ts 31-34
    using REAL metapath adjacencies
  * Focal loss, AdamW, warmup + cosine schedule
  * Fixed 400-epoch budget, best-validation-checkpoint selection
    (early stopping disabled)
  * Seeds: 0, 1, 7, 42, 123, freshly set before model construction

Because every variant sees the same seeds and the same data, the seeds are
matched and the downstream significance analysis uses PAIRED tests.

Outputs: results/component_ablation.json and results/component_ablation.csv
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from src.models.riskmagnn import create_riskmagnn
from run_multiseed_all import EPOCHS, NO_EARLY_STOP, DEVICE, set_seed, prepare_data
from run_riskmagnn import train_model, evaluate

# Same 10 seeds as run_multiseed_extended.py: the original five plus five
# fixed extensions, so every HBTBD table shares one seed set.
SEEDS = [0, 1, 7, 42, 123, 11, 21, 77, 2024, 31337]

# 2x2 design on the RiskMAGNN (Base) skeleton.
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.35

VARIANTS = [
    {'name': 'Full (TransE + RiskBias)', 'use_transe': True,  'use_risk_bias': True},
    {'name': '-TransE',                  'use_transe': False, 'use_risk_bias': True},
    {'name': '-RiskBias',                'use_transe': True,  'use_risk_bias': False},
    {'name': '-Both',                    'use_transe': False, 'use_risk_bias': False},
]


def run_one(variant: dict, seed: int, data: dict) -> dict:
    print("=" * 70)
    print(f"  {variant['name']} | seed {seed}")
    print("=" * 70)

    set_seed(seed)
    model = create_riskmagnn(
        num_features=data['train_features'].shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_metapaths=3,
        dropout=DROPOUT,
        use_transe=variant['use_transe'],
        use_risk_bias=variant['use_risk_bias'],
    ).to(DEVICE)

    model, val_pr = train_model(
        model,
        data['train_features'], data['train_labels'], data['train_adj'],
        data['val_features'], data['val_labels'], data['val_adj'],
        epochs=EPOCHS, patience=NO_EARLY_STOP,
        name=f"{variant['name']} seed={seed}",
    )

    m = evaluate(model, data['test_features'], data['test_labels'], data['test_adj'])
    print(f"  -> Val PR {val_pr:.4f} | Test PR {m['pr_auc']:.4f} | "
          f"ROC {m['roc_auc']:.4f} | F1 {m['f1']:.4f}\n")

    return {
        'variant': variant['name'],
        'use_transe': variant['use_transe'],
        'use_risk_bias': variant['use_risk_bias'],
        'seed': seed,
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
    for variant in VARIANTS:
        name = variant['name']
        rs = [r for r in records if r['variant'] == name]
        out[name] = {}
        for key in ['test_pr_auc', 'test_roc_auc', 'test_f1']:
            mean, std = stats([r[key] for r in rs])
            out[name][key] = {'mean': mean, 'std': std}
    return out


def main():
    print("=" * 70)
    print("  MULTI-SEED COMPONENT ABLATION -- Official HBTBD Split")
    print(f"  Skeleton: RiskMAGNN Base (d={HIDDEN_DIM}, L={NUM_LAYERS}, "
          f"dropout={DROPOUT})")
    print(f"  Variants: {[v['name'] for v in VARIANTS]}")
    print(f"  Seeds: {SEEDS} | Fixed {EPOCHS}-epoch budget, best-val checkpoint")
    print("=" * 70 + "\n")

    data = prepare_data()

    records = []
    for variant in VARIANTS:
        for seed in SEEDS:
            records.append(run_one(variant, seed, data))

    summary = summarize(records)

    print("\n" + "=" * 70)
    print("  PER-SEED TEST PR-AUC")
    print("=" * 70)
    header = "  {:<26}".format("Variant") + "".join(f"{s:>8}" for s in SEEDS) + f"{'mean':>9}{'std':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for variant in VARIANTS:
        name = variant['name']
        row = f"  {name:<26}"
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
    for variant in VARIANTS:
        name = variant['name']
        pr = summary[name]['test_pr_auc']
        roc = summary[name]['test_roc_auc']
        f1 = summary[name]['test_f1']
        print(f"  {name:<26} PR-AUC {pr['mean']*100:5.2f}+/-{pr['std']*100:.2f}  "
              f"ROC {roc['mean']*100:5.2f}+/-{roc['std']*100:.2f}  "
              f"F1 {f1['mean']*100:5.2f}+/-{f1['std']*100:.2f}")

    print("\n  (Run analyze_significance.py for paired tests on these seeds.)")

    os.makedirs('results', exist_ok=True)
    with open('results/component_ablation.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'epochs': EPOCHS,
                   'skeleton': {'hidden_dim': HIDDEN_DIM,
                                'num_layers': NUM_LAYERS,
                                'dropout': DROPOUT},
                   'per_run': records, 'summary': summary}, f, indent=2)
    import pandas as pd
    pd.DataFrame(records).to_csv('results/component_ablation.csv', index=False)
    print("\nSaved: results/component_ablation.json and results/component_ablation.csv")


if __name__ == "__main__":
    main()
