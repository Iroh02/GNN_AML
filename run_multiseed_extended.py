"""
Extended Multi-Seed Evaluation (n=10) with Ensembling and Tuned Thresholds
==========================================================================

Extends run_multiseed_all.py in three pre-registered, legitimate ways:

1. TEN seeds instead of five. The first five are identical to the original
   run (0, 1, 7, 42, 123) so paired comparisons with the n=5 ablation remain
   possible; five new seeds (11, 21, 77, 2024, 31337) were fixed before any
   result was observed. All models get all seeds; results are reported
   regardless of outcome.

2. SEED ENSEMBLE. For each model, the softmax probabilities of the 10
   per-seed models are averaged and evaluated once. This is a standard
   variance-reduction technique and is reported explicitly as an ensemble,
   never blended with single-model rows.

3. VALIDATION-TUNED DECISION THRESHOLD for F1. The default argmax cutoff is
   arbitrary under class imbalance. For each run, the threshold that
   maximises F1 on the VALIDATION set is selected and then applied unchanged
   to the test set. Test data is never used for tuning.

Protocol otherwise identical to run_multiseed_all.py (StandardScaler fit on
train, temporal val split ts>=31 with real subgraph adjacencies, focal loss,
AdamW, warmup+cosine, fixed 400-epoch budget, best-val checkpoint).

Outputs:
  results/multiseed_extended.json / .csv   per-run metrics (n=10)
  results/probs/<model>_seed<k>.npz        val/test probabilities per run
  results/ensemble_results.json            ensemble metrics per model
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)

from run_multiseed_all import (
    MODEL_CONFIGS, EPOCHS, NO_EARLY_STOP, DEVICE, set_seed, prepare_data,
)
from run_riskmagnn import train_model

SEEDS = [0, 1, 7, 42, 123, 11, 21, 77, 2024, 31337]
PROBS_DIR = 'results/probs'


def slug(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower()


def predict_probs(model, features, adj):
    model.eval()
    with torch.no_grad():
        out = model(features, adj)
        return F.softmax(out, dim=1)[:, 1].cpu().numpy()


def best_f1_threshold(y_true, probs):
    """Threshold maximising F1 on (y_true, probs). Never call with test data."""
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # precision_recall_curve returns len(thr) = len(prec) - 1
    best = int(np.nanargmax(f1[:-1]))
    return float(thr[best])


def metrics_at(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        'pr_auc': float(average_precision_score(y_true, probs)),
        'roc_auc': float(roc_auc_score(y_true, probs)),
        'f1': float(f1_score(y_true, preds, zero_division=0)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'threshold': float(threshold),
    }


def main():
    print("=" * 70)
    print("  EXTENDED MULTI-SEED EVALUATION (n=10) -- Official HBTBD Split")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70 + "\n")

    data = prepare_data()
    os.makedirs(PROBS_DIR, exist_ok=True)

    y_val = data['val_labels'].cpu().numpy()
    y_test = data['test_labels'].cpu().numpy()

    records = []
    for cfg in MODEL_CONFIGS:
        for seed in SEEDS:
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

            val_probs = predict_probs(model, data['val_features'], data['val_adj'])
            test_probs = predict_probs(model, data['test_features'], data['test_adj'])
            np.savez(os.path.join(PROBS_DIR, f"{slug(cfg['name'])}_seed{seed}.npz"),
                     val_probs=val_probs, test_probs=test_probs)

            m_argmax = metrics_at(y_test, test_probs, 0.5)
            thr = best_f1_threshold(y_val, val_probs)
            m_tuned = metrics_at(y_test, test_probs, thr)

            rec = {
                'model': cfg['name'], 'seed': seed,
                'val_pr_auc': float(val_pr),
                'test_pr_auc': m_argmax['pr_auc'],
                'test_roc_auc': m_argmax['roc_auc'],
                'test_f1_argmax': m_argmax['f1'],
                'test_f1_tuned': m_tuned['f1'],
                'test_precision_tuned': m_tuned['precision'],
                'test_recall_tuned': m_tuned['recall'],
                'tuned_threshold': thr,
            }
            records.append(rec)
            print(f"  -> Test PR {rec['test_pr_auc']:.4f} | ROC {rec['test_roc_auc']:.4f} "
                  f"| F1(argmax) {rec['test_f1_argmax']:.4f} "
                  f"| F1(val-tuned thr={thr:.3f}) {rec['test_f1_tuned']:.4f}\n")

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def stats(vals):
        v = np.asarray(vals, dtype=float)
        return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    summary = {}
    for cfg in MODEL_CONFIGS:
        rs = [r for r in records if r['model'] == cfg['name']]
        summary[cfg['name']] = {
            k: dict(zip(('mean', 'std'), stats([r[k] for r in rs])))
            for k in ['test_pr_auc', 'test_roc_auc', 'test_f1_argmax', 'test_f1_tuned']
        }

    print("\n" + "=" * 70)
    print("  PER-MODEL SUMMARY (mean +/- std over 10 seeds)")
    print("=" * 70)
    for name, s in summary.items():
        print(f"  {name:<20} PR {s['test_pr_auc']['mean']*100:5.2f}+/-{s['test_pr_auc']['std']*100:.2f}  "
              f"ROC {s['test_roc_auc']['mean']*100:5.2f}+/-{s['test_roc_auc']['std']*100:.2f}  "
              f"F1(argmax) {s['test_f1_argmax']['mean']*100:5.2f}+/-{s['test_f1_argmax']['std']*100:.2f}  "
              f"F1(tuned) {s['test_f1_tuned']['mean']*100:5.2f}+/-{s['test_f1_tuned']['std']*100:.2f}")

    # Seed ensembles: average probabilities across the 10 runs of each model
    ensembles = {}
    print("\n" + "=" * 70)
    print("  SEED ENSEMBLES (mean of 10 models' probabilities)")
    print("=" * 70)
    for cfg in MODEL_CONFIGS:
        vp, tp = [], []
        for seed in SEEDS:
            z = np.load(os.path.join(PROBS_DIR, f"{slug(cfg['name'])}_seed{seed}.npz"))
            vp.append(z['val_probs'])
            tp.append(z['test_probs'])
        vp, tp = np.mean(vp, axis=0), np.mean(tp, axis=0)

        thr = best_f1_threshold(y_val, vp)
        m = metrics_at(y_test, tp, thr)
        m_argmax = metrics_at(y_test, tp, 0.5)
        ensembles[cfg['name']] = {'tuned': m, 'argmax': m_argmax}
        print(f"  {cfg['name']:<20} PR {m['pr_auc']*100:5.2f}  ROC {m['roc_auc']*100:5.2f}  "
              f"F1(argmax) {m_argmax['f1']*100:5.2f}  F1(tuned thr={thr:.3f}) {m['f1']*100:5.2f}")

    os.makedirs('results', exist_ok=True)
    with open('results/multiseed_extended.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'epochs': EPOCHS,
                   'per_run': records, 'summary': summary}, f, indent=2)
    import pandas as pd
    pd.DataFrame(records).to_csv('results/multiseed_extended.csv', index=False)
    with open('results/ensemble_results.json', 'w') as f:
        json.dump(ensembles, f, indent=2)
    print("\nSaved: results/multiseed_extended.json/.csv, results/ensemble_results.json")


if __name__ == "__main__":
    main()
