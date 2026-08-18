"""
Embedding + Downstream Classifier Pipeline
============================================

Extracts GNN embeddings and trains RF/SVM classifiers on top.
This matches Song & Gu's evaluation methodology for fair comparison.

Song & Gu (2023) reported:
  - MAGNN + RF: Precision=92.2%, F1=73.9%
  - MAGNN + SVM: Precision=88.5%, F1=71.2%

Usage:
    python run_embedding_clf.py --model riskmagnn_large
    python run_embedding_clf.py --model all
    python run_embedding_clf.py --model riskmagnn_large --tune-threshold
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, classification_report
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from run_benchmark import (
    load_hbtbd_data, set_seed, create_model, extract_subgraph_adj,
    SPLIT_PATHS, ALL_MODELS
)


def extract_embeddings(model, features, metapath_adj):
    """Extract node embeddings from a trained GNN model."""
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(features, metapath_adj)
    return embeddings.cpu().numpy()


def train_and_evaluate_clf(train_emb, train_labels, test_emb, test_labels,
                           clf_name='rf', seed=42):
    """Train a downstream classifier on embeddings and evaluate."""
    if clf_name == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=seed, n_jobs=-1
        )
    elif clf_name == 'svm':
        clf = SVC(
            kernel='rbf', C=10.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=seed
        )
    elif clf_name == 'gbdt':
        # Compute scale_pos_weight for imbalance
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        spw = n_neg / max(n_pos, 1)
        sample_weight = np.where(train_labels == 1, spw, 1.0)
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=seed
        )
        clf.fit(train_emb, train_labels, sample_weight=sample_weight)
        preds = clf.predict(test_emb)
        probs = clf.predict_proba(test_emb)[:, 1]
        return _compute_metrics(test_labels, preds, probs)
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

    clf.fit(train_emb, train_labels)
    preds = clf.predict(test_emb)
    probs = clf.predict_proba(test_emb)[:, 1] if hasattr(clf, 'predict_proba') else None

    return _compute_metrics(test_labels, preds, probs)


def _compute_metrics(targets, preds, probs=None):
    metrics = {
        'precision': float(precision_score(targets, preds, zero_division=0)),
        'recall': float(recall_score(targets, preds, zero_division=0)),
        'f1': float(f1_score(targets, preds, zero_division=0)),
    }
    if probs is not None and len(np.unique(targets)) >= 2:
        metrics['pr_auc'] = float(average_precision_score(targets, probs))
        metrics['roc_auc'] = float(roc_auc_score(targets, probs))
    return metrics


def threshold_search(probs, targets, thresholds=None):
    """Search for optimal classification threshold to maximize F1."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    best_f1 = 0
    best_thresh = 0.5
    results = []

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        results.append({'threshold': float(thresh), 'f1': float(f1),
                        'precision': float(prec), 'recall': float(rec)})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1, results


def run_embedding_pipeline(model_name, seeds, split_name, device,
                           classifiers=('rf', 'svm', 'gbdt'), tune_threshold=False):
    """Run the full embedding + classifier pipeline."""
    paths = SPLIT_PATHS[split_name]

    print(f"\n{'='*70}")
    print(f"  Embedding + Classifier: {model_name} on {split_name}")
    print(f"  Classifiers: {classifiers}")
    print(f"{'='*70}")

    # Load data
    train_features, train_labels, train_timesteps, train_metapath_adj, scaler = \
        load_hbtbd_data(paths['train'], normalize=True, device=device)
    test_features, test_labels, test_timesteps, test_metapath_adj, _ = \
        load_hbtbd_data(paths['test'], normalize=True, scaler=scaler, device=device)

    if (train_labels == 1).float().mean() > 0.5:
        train_labels = 1 - train_labels
        test_labels = 1 - test_labels

    num_features = train_features.shape[1]
    all_results = {}

    for seed in seeds:
        set_seed(seed)
        print(f"\n  Seed {seed}:")

        # Train GNN model (use full training data, split val internally)
        train_mask = train_timesteps <= 30
        val_mask = train_timesteps >= 31
        train_idx = torch.where(train_mask)[0]
        val_idx = torch.where(val_mask)[0]

        if len(val_idx) == 0:
            n = len(train_labels)
            perm = torch.randperm(n)
            split = int(0.8 * n)
            train_idx = perm[:split]
            val_idx = perm[split:]

        actual_train_feat = train_features[train_idx]
        actual_train_lab = train_labels[train_idx]
        actual_train_adj = [extract_subgraph_adj(adj, train_idx) for adj in train_metapath_adj]
        val_feat = train_features[val_idx]
        val_lab = train_labels[val_idx]
        val_adj = [extract_subgraph_adj(adj, val_idx) for adj in train_metapath_adj]

        # Create and train GNN
        model = create_model(model_name, num_features, device=device)
        from run_benchmark import train_model
        model, _ = train_model(
            model, actual_train_feat, actual_train_lab, actual_train_adj,
            val_feat, val_lab, val_adj,
            epochs=350, patience=50, verbose=False
        )

        # Extract embeddings (full train + test)
        train_emb = extract_embeddings(model, train_features, train_metapath_adj)
        test_emb = extract_embeddings(model, test_features, test_metapath_adj)
        train_lab_np = train_labels.cpu().numpy()
        test_lab_np = test_labels.cpu().numpy()

        # Also get end-to-end GNN results for comparison
        model.eval()
        with torch.no_grad():
            out = model(test_features, test_metapath_adj)
            gnn_probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            gnn_preds = out.argmax(dim=1).cpu().numpy()

        gnn_metrics = _compute_metrics(test_lab_np, gnn_preds, gnn_probs)

        if seed not in all_results:
            all_results[seed] = {'end_to_end': gnn_metrics}

        # Threshold search for end-to-end
        if tune_threshold:
            best_thresh, best_f1, _ = threshold_search(gnn_probs, test_lab_np)
            tuned_preds = (gnn_probs >= best_thresh).astype(int)
            tuned_metrics = _compute_metrics(test_lab_np, tuned_preds, gnn_probs)
            tuned_metrics['threshold'] = float(best_thresh)
            all_results[seed]['end_to_end_tuned'] = tuned_metrics
            print(f"    End-to-end (tuned @{best_thresh:.2f}): "
                  f"F1={tuned_metrics['f1']*100:.1f}%, Prec={tuned_metrics['precision']*100:.1f}%")

        # Train downstream classifiers
        for clf_name in classifiers:
            print(f"    {clf_name.upper()}: ", end='')
            metrics = train_and_evaluate_clf(
                train_emb, train_lab_np, test_emb, test_lab_np, clf_name, seed
            )
            all_results[seed][clf_name] = metrics
            print(f"F1={metrics['f1']*100:.1f}%, Prec={metrics['precision']*100:.1f}%, "
                  f"PR-AUC={metrics.get('pr_auc', 0)*100:.1f}%")

    # Aggregate across seeds
    print(f"\n  {'='*60}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"  {'='*60}")
    print(f"  {'Method':<25s} {'F1':>15s} {'Precision':>15s} {'PR-AUC':>15s}")
    print(f"  {'-'*70}")

    aggregate = {}
    methods = ['end_to_end'] + list(classifiers)
    if tune_threshold:
        methods.insert(1, 'end_to_end_tuned')

    for method in methods:
        f1s = [all_results[s][method]['f1'] for s in seeds if method in all_results[s]]
        precs = [all_results[s][method]['precision'] for s in seeds if method in all_results[s]]
        pr_aucs = [all_results[s][method].get('pr_auc', 0) for s in seeds if method in all_results[s]]

        if not f1s:
            continue

        f1_str = f"{np.mean(f1s)*100:.1f} +/- {np.std(f1s)*100:.1f}%"
        prec_str = f"{np.mean(precs)*100:.1f} +/- {np.std(precs)*100:.1f}%"
        pr_str = f"{np.mean(pr_aucs)*100:.1f} +/- {np.std(pr_aucs)*100:.1f}%"
        print(f"  {method:<25s} {f1_str:>15s} {prec_str:>15s} {pr_str:>15s}")

        aggregate[method] = {
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'precision_mean': float(np.mean(precs)), 'precision_std': float(np.std(precs)),
            'pr_auc_mean': float(np.mean(pr_aucs)), 'pr_auc_std': float(np.std(pr_aucs)),
        }

    # Song & Gu comparison
    print(f"\n  {'='*60}")
    print(f"  COMPARISON WITH SONG & GU (2023)")
    print(f"  {'='*60}")
    print(f"  {'Song & Gu MAGNN+RF':<25s} {'73.9%':>15s} {'92.2%':>15s} {'N/A':>15s}")
    print(f"  {'Song & Gu MAGNN+SVM':<25s} {'71.2%':>15s} {'88.5%':>15s} {'N/A':>15s}")

    # Save results
    os.makedirs('results/benchmark', exist_ok=True)
    output = {
        'model': model_name,
        'split': split_name,
        'seeds': seeds,
        'per_seed': {str(k): v for k, v in all_results.items()},
        'aggregate': aggregate,
    }
    fname = f'results/benchmark/{model_name}_embedding_clf_{split_name}.json'
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {fname}")

    return output


def main():
    parser = argparse.ArgumentParser(description='Embedding + Classifier Pipeline')
    parser.add_argument('--model', type=str, default='riskmagnn_large',
                        help='GNN model for embedding extraction')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--split', type=str, default='official',
                        choices=list(SPLIT_PATHS.keys()))
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--tune-threshold', action='store_true',
                        help='Search for optimal classification threshold')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = 'cpu'

    models = ALL_MODELS if args.model == 'all' else [args.model]

    for model_name in models:
        run_embedding_pipeline(
            model_name, args.seeds, args.split, device,
            tune_threshold=args.tune_threshold
        )


if __name__ == "__main__":
    main()
