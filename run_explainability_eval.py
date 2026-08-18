"""
Quantitative Explainability Evaluation
========================================

Evaluates GNNExplainer with quantitative fidelity metrics,
scaled to 100+ nodes with statistical aggregation.

Metrics:
- Fidelity+ (Sufficiency): Does keeping top-k edges preserve the prediction?
- Fidelity- (Necessity): Does removing top-k edges degrade the prediction?
- Sparsity: How sparse are the explanations?
- Attention Analysis: Compare metapath attention for illicit vs licit nodes

Usage:
    python run_explainability_eval.py --num-nodes 100
    python run_explainability_eval.py --num-nodes 50 --device cuda
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from run_benchmark import (
    load_hbtbd_data, set_seed, create_model, extract_subgraph_adj,
    train_model, evaluate, SPLIT_PATHS
)


# =========================================================================== #
#  Fidelity Metrics
# =========================================================================== #

def compute_fidelity_plus(model, features, metapath_adj, node_idx, edge_masks,
                          feature_mask, top_k_edges=0.3, top_k_features=0.3):
    """
    Fidelity+ (Sufficiency): Keep only the important edges/features.
    Measures how well the explanation preserves the original prediction.

    fidelity+ = P(y|full_graph) - P(y|important_only)
    Lower is better (important subgraph preserves prediction).
    """
    model.eval()
    with torch.no_grad():
        # Original prediction
        orig_logits = model(features, metapath_adj)
        orig_prob = F.softmax(orig_logits[node_idx], dim=0)
        orig_pred = orig_prob.argmax().item()
        orig_conf = orig_prob[orig_pred].item()

        # Apply masks: keep only top-k important edges
        masked_adj = _apply_top_k_edge_mask(metapath_adj, node_idx, edge_masks, top_k_edges, keep=True)
        masked_features = _apply_top_k_feature_mask(features, feature_mask, top_k_features, keep=True)

        # Prediction with only important parts
        masked_logits = model(masked_features, masked_adj)
        masked_prob = F.softmax(masked_logits[node_idx], dim=0)
        masked_conf = masked_prob[orig_pred].item()

    return orig_conf - masked_conf  # Lower = better explanation


def compute_fidelity_minus(model, features, metapath_adj, node_idx, edge_masks,
                           feature_mask, top_k_edges=0.3, top_k_features=0.3):
    """
    Fidelity- (Necessity): Remove the important edges/features.
    Measures how much the prediction degrades when explanation is removed.

    fidelity- = P(y|full_graph) - P(y|without_important)
    Higher is better (removing important parts hurts prediction).
    """
    model.eval()
    with torch.no_grad():
        orig_logits = model(features, metapath_adj)
        orig_prob = F.softmax(orig_logits[node_idx], dim=0)
        orig_pred = orig_prob.argmax().item()
        orig_conf = orig_prob[orig_pred].item()

        # Remove top-k important edges
        masked_adj = _apply_top_k_edge_mask(metapath_adj, node_idx, edge_masks, top_k_edges, keep=False)
        masked_features = _apply_top_k_feature_mask(features, feature_mask, top_k_features, keep=False)

        masked_logits = model(masked_features, masked_adj)
        masked_prob = F.softmax(masked_logits[node_idx], dim=0)
        masked_conf = masked_prob[orig_pred].item()

    return orig_conf - masked_conf  # Higher = better explanation


def _apply_top_k_edge_mask(metapath_adj, node_idx, edge_masks, top_k, keep=True):
    """Apply top-k edge mask. If keep=True, keep only top-k edges; else remove them."""
    masked_adj = []
    for mp_idx, (adj, emask) in enumerate(zip(metapath_adj, edge_masks)):
        indices = adj._indices()
        values = adj._values().clone()
        dst = indices[1]

        node_edges = (dst == node_idx)
        if node_edges.sum() == 0 or len(emask) == 0:
            masked_adj.append(adj)
            continue

        num_edges = node_edges.sum().item()
        k = max(1, int(top_k * num_edges))

        # Get importance order
        importance = emask[:num_edges] if len(emask) >= num_edges else emask
        if len(importance) < num_edges:
            importance = np.pad(importance, (0, num_edges - len(importance)), constant_values=0.5)

        threshold_idx = min(k, len(importance))
        sorted_imp = np.sort(importance)[::-1]
        threshold = sorted_imp[threshold_idx - 1] if threshold_idx > 0 else 0

        edge_indices = torch.where(node_edges)[0]
        for i, idx in enumerate(edge_indices):
            if i < len(importance):
                if keep:
                    values[idx] = values[idx] if importance[i] >= threshold else 0.0
                else:
                    values[idx] = 0.0 if importance[i] >= threshold else values[idx]

        new_adj = torch.sparse_coo_tensor(indices, values, adj.shape).coalesce()
        masked_adj.append(new_adj)

    return masked_adj


def _apply_top_k_feature_mask(features, feature_mask, top_k, keep=True):
    """Apply top-k feature mask."""
    k = max(1, int(top_k * len(feature_mask)))
    sorted_imp = np.sort(feature_mask)[::-1]
    threshold = sorted_imp[k - 1] if k > 0 else 0

    mask = torch.ones(features.shape[1], device=features.device)
    for i, imp in enumerate(feature_mask):
        if keep:
            mask[i] = 1.0 if imp >= threshold else 0.0
        else:
            mask[i] = 0.0 if imp >= threshold else 1.0

    return features * mask.unsqueeze(0)


# =========================================================================== #
#  Lightweight Gradient-based Explanation (faster than GNNExplainer)
# =========================================================================== #

def gradient_explanation(model, features, metapath_adj, node_idx):
    """
    Fast gradient-based explanation for a single node.
    Returns edge importance per metapath and feature importance.
    """
    model.eval()
    features_grad = features.detach().clone().requires_grad_(True)

    # Forward pass
    logits = model(features_grad, metapath_adj)
    pred = logits[node_idx].argmax().item()
    conf = F.softmax(logits[node_idx].detach(), dim=0)[pred].item()

    # Backward for feature importance
    logits[node_idx, pred].backward(retain_graph=True)
    feature_importance = features_grad.grad[node_idx].abs().cpu().numpy()

    # Edge importance via perturbation (per metapath)
    edge_masks = []
    for mp_idx, adj in enumerate(metapath_adj):
        indices = adj._indices()
        dst = indices[1]
        node_edges = (dst == node_idx)

        if node_edges.sum() == 0:
            edge_masks.append(np.array([]))
            continue

        edge_indices = torch.where(node_edges)[0]
        importances = []

        with torch.no_grad():
            for eidx in edge_indices[:50]:  # Cap at 50 edges for speed
                # Remove this edge and measure prediction change
                new_values = adj._values().clone()
                new_values[eidx] = 0.0
                new_adj = [a if i != mp_idx else
                           torch.sparse_coo_tensor(adj._indices(), new_values, adj.shape).coalesce()
                           for i, a in enumerate(metapath_adj)]
                new_logits = model(features, new_adj)
                new_conf = F.softmax(new_logits[node_idx], dim=0)[pred].item()
                importances.append(conf - new_conf)  # Drop in confidence

        edge_masks.append(np.array(importances))

    return {
        'node_idx': node_idx,
        'prediction': pred,
        'confidence': conf,
        'feature_importance': feature_importance,
        'edge_masks': edge_masks,
    }


# =========================================================================== #
#  Attention Analysis
# =========================================================================== #

def analyze_attention(model, features, metapath_adj, labels):
    """
    Analyze metapath attention distributions for illicit vs licit nodes.
    Returns statistical test results.
    """
    model.eval()
    with torch.no_grad():
        attn_weights = model.get_attention_weights(features, metapath_adj)

    results = {}
    labels_np = labels.cpu().numpy()
    illicit_mask = labels_np == 1
    licit_mask = labels_np == 0

    for layer_idx, attn in enumerate(attn_weights):
        attn_np = attn.cpu().numpy()  # (N, num_metapaths)
        layer_results = {}

        for mp_idx, mp_name in enumerate(['M1_centralized', 'M2_oneway', 'M3_decentralized']):
            illicit_attn = attn_np[illicit_mask, mp_idx]
            licit_attn = attn_np[licit_mask, mp_idx]

            # Mann-Whitney U test
            if len(illicit_attn) > 0 and len(licit_attn) > 0:
                stat, pvalue = stats.mannwhitneyu(illicit_attn, licit_attn, alternative='two-sided')
                effect_size = abs(np.mean(illicit_attn) - np.mean(licit_attn))
            else:
                stat, pvalue, effect_size = 0, 1.0, 0

            layer_results[mp_name] = {
                'illicit_mean': float(np.mean(illicit_attn)) if len(illicit_attn) > 0 else 0,
                'illicit_std': float(np.std(illicit_attn)) if len(illicit_attn) > 0 else 0,
                'licit_mean': float(np.mean(licit_attn)) if len(licit_attn) > 0 else 0,
                'licit_std': float(np.std(licit_attn)) if len(licit_attn) > 0 else 0,
                'mann_whitney_U': float(stat),
                'p_value': float(pvalue),
                'effect_size': float(effect_size),
                'significant': bool(pvalue < 0.05),
            }

        results[f'layer_{layer_idx}'] = layer_results

    return results


# =========================================================================== #
#  Feature Importance Aggregation
# =========================================================================== #

def aggregate_feature_importance(all_explanations):
    """Aggregate feature importance across all explained nodes."""
    all_importances = np.stack([e['feature_importance'] for e in all_explanations])
    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)

    # Group by feature category
    num_features = len(mean_importance)
    categories = {
        'local (0-40)': list(range(min(41, num_features))),
        '1-hop (41-102)': list(range(41, min(103, num_features))),
        '2-hop (103-164)': list(range(103, min(165, num_features))),
    }

    category_stats = {}
    for cat_name, indices in categories.items():
        if not indices:
            continue
        cat_imp = mean_importance[indices]
        category_stats[cat_name] = {
            'mean_importance': float(cat_imp.mean()),
            'max_importance': float(cat_imp.max()),
            'count_in_top10': int(sum(1 for i in np.argsort(mean_importance)[-10:] if i in indices)),
        }

    return {
        'mean_importance': mean_importance.tolist(),
        'std_importance': std_importance.tolist(),
        'top_10_features': [int(i) for i in np.argsort(mean_importance)[-10:][::-1]],
        'category_stats': category_stats,
    }


# =========================================================================== #
#  Main Pipeline
# =========================================================================== #

def run_explainability_eval(num_nodes=100, split_name='official', device='cpu', seed=42):
    """Run full quantitative explainability evaluation."""
    set_seed(seed)
    paths = SPLIT_PATHS[split_name]

    print(f"\n{'='*70}")
    print(f"  Quantitative Explainability Evaluation")
    print(f"  Nodes: {num_nodes} | Split: {split_name} | Device: {device}")
    print(f"{'='*70}")

    # Load data
    print("\n  Loading data...")
    train_features, train_labels, train_timesteps, train_metapath_adj, scaler = \
        load_hbtbd_data(paths['train'], normalize=True, device=device)
    test_features, test_labels, test_timesteps, test_metapath_adj, _ = \
        load_hbtbd_data(paths['test'], normalize=True, scaler=scaler, device=device)

    if (train_labels == 1).float().mean() > 0.5:
        train_labels = 1 - train_labels
        test_labels = 1 - test_labels

    # Train model
    print("  Training RiskMAGNN (Large)...")
    num_features = train_features.shape[1]
    train_mask = train_timesteps <= 30
    val_mask = train_timesteps >= 31
    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]

    if len(val_idx) == 0:
        n = len(train_labels)
        perm = torch.randperm(n)
        split = int(0.8 * n)
        train_idx, val_idx = perm[:split], perm[split:]

    model = create_model('riskmagnn_large', num_features, device=device)
    model, _ = train_model(
        model, train_features[train_idx], train_labels[train_idx],
        [extract_subgraph_adj(a, train_idx) for a in train_metapath_adj],
        train_features[val_idx], train_labels[val_idx],
        [extract_subgraph_adj(a, val_idx) for a in train_metapath_adj],
        epochs=350, patience=50, verbose=True
    )

    # Get predictions on test set
    print("\n  Getting predictions...")
    test_metrics = evaluate(model, test_features, test_labels, test_metapath_adj)
    print(f"  Test PR-AUC: {test_metrics['pr_auc']:.4f}, F1: {test_metrics['f1']:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(test_features, test_metapath_adj)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    test_labels_np = test_labels.cpu().numpy()

    # Select nodes for explanation (balanced: TP, FP, TN)
    tp_idx = np.where((preds == 1) & (test_labels_np == 1))[0]
    fp_idx = np.where((preds == 1) & (test_labels_np == 0))[0]
    tn_idx = np.where((preds == 0) & (test_labels_np == 0))[0]

    n_per_group = num_nodes // 3
    np.random.shuffle(tp_idx)
    np.random.shuffle(fp_idx)
    np.random.shuffle(tn_idx)

    selected_tp = tp_idx[:n_per_group]
    selected_fp = fp_idx[:n_per_group]
    selected_tn = tn_idx[:n_per_group]

    print(f"\n  Selected nodes: {len(selected_tp)} TP, {len(selected_fp)} FP, {len(selected_tn)} TN")

    # Run gradient-based explanations (fast)
    print("\n  Computing gradient-based explanations...")
    all_explanations = []
    all_fidelity_plus = {'tp': [], 'fp': [], 'tn': []}
    all_fidelity_minus = {'tp': [], 'fp': [], 'tn': []}
    all_sparsity = []

    groups = [
        ('tp', selected_tp),
        ('fp', selected_fp),
        ('tn', selected_tn),
    ]

    for group_name, node_indices in groups:
        print(f"    {group_name.upper()}: explaining {len(node_indices)} nodes...")
        for i, nidx in enumerate(node_indices):
            if (i + 1) % 10 == 0:
                print(f"      {i+1}/{len(node_indices)}")

            explanation = gradient_explanation(model, test_features, test_metapath_adj, nidx)
            all_explanations.append(explanation)

            # Compute fidelity metrics
            fp = compute_fidelity_plus(
                model, test_features, test_metapath_adj, nidx,
                explanation['edge_masks'], explanation['feature_importance']
            )
            fm = compute_fidelity_minus(
                model, test_features, test_metapath_adj, nidx,
                explanation['edge_masks'], explanation['feature_importance']
            )
            all_fidelity_plus[group_name].append(fp)
            all_fidelity_minus[group_name].append(fm)

            # Sparsity: fraction of non-zero edge importances
            total_edges = sum(len(em) for em in explanation['edge_masks'])
            important_edges = sum(
                np.sum(np.abs(em) > 0.01) for em in explanation['edge_masks']
            )
            sparsity = 1.0 - (important_edges / max(total_edges, 1))
            all_sparsity.append(sparsity)

    # Aggregate fidelity metrics
    print(f"\n  {'='*60}")
    print(f"  FIDELITY RESULTS")
    print(f"  {'='*60}")
    print(f"  {'Group':<8s} {'Fidelity+ (lower=better)':>25s} {'Fidelity- (higher=better)':>25s}")
    print(f"  {'-'*60}")

    fidelity_summary = {}
    for group in ['tp', 'fp', 'tn']:
        fp_vals = all_fidelity_plus[group]
        fm_vals = all_fidelity_minus[group]
        if fp_vals:
            fp_str = f"{np.mean(fp_vals):.4f} +/- {np.std(fp_vals):.4f}"
            fm_str = f"{np.mean(fm_vals):.4f} +/- {np.std(fm_vals):.4f}"
            print(f"  {group.upper():<8s} {fp_str:>25s} {fm_str:>25s}")
            fidelity_summary[group] = {
                'fidelity_plus_mean': float(np.mean(fp_vals)),
                'fidelity_plus_std': float(np.std(fp_vals)),
                'fidelity_minus_mean': float(np.mean(fm_vals)),
                'fidelity_minus_std': float(np.std(fm_vals)),
            }

    print(f"\n  Sparsity: {np.mean(all_sparsity):.4f} +/- {np.std(all_sparsity):.4f}")

    # Attention analysis
    print(f"\n  {'='*60}")
    print(f"  ATTENTION ANALYSIS (illicit vs licit)")
    print(f"  {'='*60}")

    attn_results = analyze_attention(model, test_features, test_metapath_adj, test_labels)

    for layer_name, layer_data in attn_results.items():
        print(f"\n  {layer_name}:")
        for mp_name, mp_data in layer_data.items():
            sig = "*" if mp_data['significant'] else ""
            print(f"    {mp_name}: illicit={mp_data['illicit_mean']:.4f}, "
                  f"licit={mp_data['licit_mean']:.4f}, "
                  f"p={mp_data['p_value']:.4f}{sig}")

    # Feature importance aggregation
    print(f"\n  {'='*60}")
    print(f"  FEATURE IMPORTANCE BY CATEGORY")
    print(f"  {'='*60}")

    feat_agg = aggregate_feature_importance(all_explanations)
    for cat_name, cat_data in feat_agg['category_stats'].items():
        print(f"  {cat_name}: mean_imp={cat_data['mean_importance']:.4f}, "
              f"in_top10={cat_data['count_in_top10']}")

    # Save all results
    results = {
        'test_metrics': test_metrics,
        'num_nodes_explained': len(all_explanations),
        'groups': {'tp': len(selected_tp), 'fp': len(selected_fp), 'tn': len(selected_tn)},
        'fidelity': fidelity_summary,
        'sparsity_mean': float(np.mean(all_sparsity)),
        'sparsity_std': float(np.std(all_sparsity)),
        'attention_analysis': attn_results,
        'feature_importance': feat_agg,
    }

    os.makedirs('results/benchmark', exist_ok=True)
    fname = f'results/benchmark/explainability_eval_{split_name}.json'
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {fname}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Quantitative Explainability Evaluation')
    parser.add_argument('--num-nodes', type=int, default=100,
                        help='Number of nodes to explain (split into TP/FP/TN)')
    parser.add_argument('--split', type=str, default='official',
                        choices=list(SPLIT_PATHS.keys()))
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = 'cpu'

    run_explainability_eval(args.num_nodes, args.split, device, args.seed)


if __name__ == "__main__":
    main()
