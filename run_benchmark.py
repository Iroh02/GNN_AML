"""
Unified Benchmark Runner for HBTBD
====================================

Runs all models with multi-seed evaluation and GPU support.
Produces publication-ready results with mean +/- std.

Usage:
    # Run a single model with multiple seeds
    python run_benchmark.py --model riskmagnn --seeds 42 123 456 789 1024

    # Run all models
    python run_benchmark.py --model all --seeds 42 123 456 789 1024

    # Run ablation study
    python run_benchmark.py --model riskmagnn --ablation no_transe

    # Use GPU
    python run_benchmark.py --model riskmagnn --device cuda

    # Run on different split
    python run_benchmark.py --model riskmagnn --split resplit
"""

import os
import sys
import json
import time
import random
import argparse
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from src.models.riskmagnn import RiskMAGNN, create_riskmagnn
from src.models.magnn import SimpleHeterogeneousGNN, HANStyleHeteroGNN
from src.models.baselines_hbtbd import (
    GCNBaseline, GATBaseline, RGCNBaseline, HGTBaseline, create_baseline
)


# =========================================================================== #
#  Data Loading (adapted from run_riskmagnn.py)
# =========================================================================== #

def load_adjlist(filepath: str, num_nodes: int, device: str = 'cpu') -> torch.Tensor:
    """Load metapath adjacency list and convert to sparse tensor."""
    rows, cols = [], []

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            (num_nodes, num_nodes)
        ).coalesce().to(device)

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                src = int(parts[0])
                for dst in parts[1:min(len(parts), 51)]:
                    dst = int(dst)
                    if src < num_nodes and dst < num_nodes:
                        rows.append(src)
                        cols.append(dst)

    if len(rows) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            (num_nodes, num_nodes)
        ).coalesce().to(device)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce().to(device)


def load_hbtbd_data(data_path: str, normalize: bool = True, scaler=None, device: str = 'cpu'):
    """Load HBTBD dataset."""
    features = np.load(os.path.join(data_path, 'features0.npy'))
    labels = np.load(os.path.join(data_path, 'labels.npy'))
    timesteps = np.load(os.path.join(data_path, 'timesteps.npy'))
    node_types = np.load(os.path.join(data_path, 'node_types.npy'))

    num_tx = (node_types == 0).sum()

    if normalize:
        if scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)

    metapath_adj = []
    for mp in ['m1', 'm2', 'm3']:
        adj_path = os.path.join(data_path, f'{mp}.adjlist')
        adj = load_adjlist(adj_path, num_tx, device)
        metapath_adj.append(adj)

    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    timesteps = torch.tensor(timesteps, dtype=torch.long).to(device)

    return features, labels, timesteps, metapath_adj, scaler


def extract_subgraph_adj(adj: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    """Extract subgraph adjacency for given node indices."""
    adj_indices = adj._indices()
    adj_values = adj._values()
    device = adj.device

    num_nodes = len(node_indices)
    old_to_new = torch.full((adj.shape[0],), -1, dtype=torch.long, device=device)
    old_to_new[node_indices] = torch.arange(num_nodes, device=device)

    src, dst = adj_indices[0], adj_indices[1]
    mask = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)

    new_src = old_to_new[src[mask]]
    new_dst = old_to_new[dst[mask]]
    new_values = adj_values[mask]

    if len(new_src) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            (num_nodes, num_nodes)
        ).coalesce()

    return torch.sparse_coo_tensor(
        torch.stack([new_src, new_dst]), new_values, (num_nodes, num_nodes)
    ).coalesce()


# =========================================================================== #
#  Training Utilities
# =========================================================================== #

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.shape[1]
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        probs = F.softmax(logits, dim=1)
        pt = (probs * smooth_targets).sum(dim=1)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        ce = -torch.sum(smooth_targets * F.log_softmax(logits, dim=1), dim=1)
        return (focal_weight * ce).mean()


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, features, labels, metapath_adj):
    """Evaluate model and return all metrics."""
    model.eval()
    with torch.no_grad():
        out = model(features, metapath_adj)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

    if len(np.unique(targets)) < 2:
        return {'pr_auc': 0.0, 'roc_auc': 0.0, 'precision': 0.0,
                'recall': 0.0, 'f1': 0.0}

    return {
        'pr_auc': float(average_precision_score(targets, probs)),
        'roc_auc': float(roc_auc_score(targets, probs)),
        'precision': float(precision_score(targets, preds, zero_division=0)),
        'recall': float(recall_score(targets, preds, zero_division=0)),
        'f1': float(f1_score(targets, preds, zero_division=0)),
    }


def train_model(model, train_features, train_labels, train_adj,
                val_features, val_labels, val_adj,
                epochs=300, lr=0.001, patience=40, verbose=True):
    """Train a model with early stopping. Returns (model, best_val_pr_auc)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    warmup_epochs = 15

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75, smoothing=0.1)

    best_val_pr = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_features, train_adj)
        loss = loss_fn(out, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, val_features, val_labels, val_adj)
            if val_metrics['pr_auc'] > best_val_pr:
                best_val_pr = val_metrics['pr_auc']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1:3d}: loss={loss.item():.4f}, "
                      f"val_pr_auc={val_metrics['pr_auc']:.4f}, best={best_val_pr:.4f}")

            if wait >= patience // 10:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_pr


# =========================================================================== #
#  Model Factory
# =========================================================================== #

ABLATION_CONFIGS = {
    'full':           {'use_transe': True,  'use_risk_bias': True,  'use_residual': True,  'use_layernorm': True},
    'no_transe':      {'use_transe': False, 'use_risk_bias': True,  'use_residual': True,  'use_layernorm': True},
    'no_risk_bias':   {'use_transe': True,  'use_risk_bias': False, 'use_residual': True,  'use_layernorm': True},
    'no_transe_no_rb':{'use_transe': False, 'use_risk_bias': False, 'use_residual': True,  'use_layernorm': True},
    'no_residual':    {'use_transe': True,  'use_risk_bias': True,  'use_residual': False, 'use_layernorm': True},
    'no_layernorm':   {'use_transe': True,  'use_risk_bias': True,  'use_residual': True,  'use_layernorm': False},
}


def create_model(model_name: str, num_features: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 ablation: str = 'full', device: str = 'cpu'):
    """Create any model by name."""

    if model_name == 'riskmagnn':
        abl = ABLATION_CONFIGS.get(ablation, ABLATION_CONFIGS['full'])
        model = create_riskmagnn(
            num_features=num_features, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, **abl
        )
    elif model_name == 'riskmagnn_large':
        abl = ABLATION_CONFIGS.get(ablation, ABLATION_CONFIGS['full'])
        model = create_riskmagnn(
            num_features=num_features, hidden_dim=192,
            num_layers=3, dropout=0.4, **abl
        )
    elif model_name == 'simplehetero':
        model = SimpleHeterogeneousGNN(
            num_features=num_features, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout
        )
    elif model_name == 'han':
        model = HANStyleHeteroGNN(
            num_features=num_features, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout
        )
    elif model_name in ('gcn', 'gat', 'rgcn', 'hgt'):
        model = create_baseline(
            model_name, num_features=num_features,
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


ALL_MODELS = ['gcn', 'gat', 'rgcn', 'hgt', 'han', 'simplehetero', 'riskmagnn', 'riskmagnn_large']
ALL_ABLATIONS = list(ABLATION_CONFIGS.keys())


# =========================================================================== #
#  Split Paths
# =========================================================================== #

SPLIT_PATHS = {
    'official': {
        'train': 'data/hbtbd/HBTBD/data/train/',
        'test':  'data/hbtbd/HBTBD/data/test/',
    },
    'resplit': {
        'train': 'data/hbtbd_resplit/train/',
        'test':  'data/hbtbd_resplit/test/',
    },
    'chrono': {
        'train': 'data/hbtbd_chrono/train/',
        'test':  'data/hbtbd_chrono/test/',
    },
}


# =========================================================================== #
#  Main Benchmark
# =========================================================================== #

def run_single_experiment(model_name, seed, num_features, train_features, train_labels,
                          train_timesteps, train_metapath_adj, test_features, test_labels,
                          test_metapath_adj, device, ablation='full', verbose=True):
    """Run a single experiment with a given seed. Returns metrics dict."""
    set_seed(seed)

    # Create temporal val split (train: ts 1-30, val: ts 31+)
    train_mask = train_timesteps <= 30
    val_mask = train_timesteps >= 31

    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]

    if len(val_idx) == 0:
        # Fallback: random 80/20 split if no temporal info
        n = len(train_labels)
        perm = torch.randperm(n)
        split = int(0.8 * n)
        train_idx = perm[:split]
        val_idx = perm[split:]

    actual_train_features = train_features[train_idx]
    actual_train_labels = train_labels[train_idx]
    actual_train_adj = [extract_subgraph_adj(adj, train_idx) for adj in train_metapath_adj]

    val_features = train_features[val_idx]
    val_labels = train_labels[val_idx]
    val_adj = [extract_subgraph_adj(adj, val_idx) for adj in train_metapath_adj]

    # Create model
    model = create_model(model_name, num_features, device=device, ablation=ablation)
    num_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"  Seed {seed}: {num_params:,} params, "
              f"train={len(actual_train_labels)}, val={len(val_labels)}")

    # Train
    t0 = time.time()
    model, val_pr = train_model(
        model, actual_train_features, actual_train_labels, actual_train_adj,
        val_features, val_labels, val_adj,
        epochs=350, patience=50, verbose=False
    )
    train_time = time.time() - t0

    # Evaluate on test
    test_metrics = evaluate(model, test_features, test_labels, test_metapath_adj)
    test_metrics['val_pr_auc'] = float(val_pr)
    test_metrics['train_time_s'] = round(train_time, 1)
    test_metrics['num_params'] = num_params
    test_metrics['seed'] = seed

    if verbose:
        print(f"         -> test PR-AUC={test_metrics['pr_auc']:.4f}, "
              f"F1={test_metrics['f1']:.4f}, time={train_time:.0f}s")

    return test_metrics, model


def run_benchmark(model_name, seeds, split_name, device, ablation='full'):
    """Run full benchmark for a model across multiple seeds."""
    paths = SPLIT_PATHS[split_name]

    print(f"\n{'='*70}")
    print(f"  Model: {model_name} | Split: {split_name} | Ablation: {ablation}")
    print(f"  Seeds: {seeds} | Device: {device}")
    print(f"{'='*70}")

    # Load data
    print("\n  Loading data...")
    train_features, train_labels, train_timesteps, train_metapath_adj, scaler = \
        load_hbtbd_data(paths['train'], normalize=True, device=device)
    test_features, test_labels, test_timesteps, test_metapath_adj, _ = \
        load_hbtbd_data(paths['test'], normalize=True, scaler=scaler, device=device)

    # Swap labels if needed (illicit should be minority class = label 1)
    if (train_labels == 1).float().mean() > 0.5:
        train_labels = 1 - train_labels
        test_labels = 1 - test_labels

    num_features = train_features.shape[1]
    print(f"  Train: {len(train_labels):,} nodes ({(train_labels==1).float().mean()*100:.1f}% illicit)")
    print(f"  Test:  {len(test_labels):,} nodes ({(test_labels==1).float().mean()*100:.1f}% illicit)")

    # Run experiments
    all_results = []
    best_model = None
    best_pr_auc = 0.0

    for seed in seeds:
        metrics, model = run_single_experiment(
            model_name, seed, num_features,
            train_features, train_labels, train_timesteps, train_metapath_adj,
            test_features, test_labels, test_metapath_adj,
            device, ablation=ablation
        )
        all_results.append(metrics)
        if metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = metrics['pr_auc']
            best_model = model

    # Compute aggregate statistics
    metric_keys = ['pr_auc', 'roc_auc', 'f1', 'precision', 'recall']
    summary = {}
    for key in metric_keys:
        values = [r[key] for r in all_results]
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))

    summary['model'] = model_name
    summary['ablation'] = ablation
    summary['split'] = split_name
    summary['seeds'] = seeds
    summary['num_params'] = all_results[0]['num_params']
    summary['per_seed'] = all_results

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  RESULTS: {model_name} ({ablation}) on {split_name}")
    print(f"  {'='*50}")
    for key in metric_keys:
        mean = summary[f'{key}_mean'] * 100
        std = summary[f'{key}_std'] * 100
        print(f"  {key:>12s}: {mean:.2f} +/- {std:.2f}%")
    print(f"  {'params':>12s}: {summary['num_params']:,}")

    # Save results
    os.makedirs('results/benchmark', exist_ok=True)
    fname = f"{model_name}_{ablation}_{split_name}_results.json"
    with open(f'results/benchmark/{fname}', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to results/benchmark/{fname}")

    # Save best model
    if best_model is not None:
        os.makedirs('models/benchmark', exist_ok=True)
        model_path = f"models/benchmark/{model_name}_{ablation}_{split_name}_best.pth"
        torch.save(best_model.state_dict(), model_path)

    return summary


def print_comparison_table(all_summaries):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"  BENCHMARK COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"  {'Model':<25s} {'PR-AUC':>15s} {'ROC-AUC':>15s} {'F1':>15s} {'Params':>10s}")
    print(f"  {'-'*80}")

    # Sort by PR-AUC
    sorted_summaries = sorted(all_summaries, key=lambda s: s['pr_auc_mean'], reverse=True)
    for s in sorted_summaries:
        name = f"{s['model']}"
        if s['ablation'] != 'full':
            name += f" ({s['ablation']})"
        pr = f"{s['pr_auc_mean']*100:.2f} +/- {s['pr_auc_std']*100:.2f}"
        roc = f"{s['roc_auc_mean']*100:.2f} +/- {s['roc_auc_std']*100:.2f}"
        f1 = f"{s['f1_mean']*100:.2f} +/- {s['f1_std']*100:.2f}"
        params = f"{s['num_params']:,}"
        print(f"  {name:<25s} {pr:>15s} {roc:>15s} {f1:>15s} {params:>10s}")

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description='HBTBD Benchmark Runner')
    parser.add_argument('--model', type=str, default='riskmagnn',
                        help=f'Model name or "all". Choices: {ALL_MODELS}')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024],
                        help='Random seeds for multi-seed evaluation')
    parser.add_argument('--split', type=str, default='official',
                        choices=list(SPLIT_PATHS.keys()),
                        help='Data split to use')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: "cpu", "cuda", or "auto"')
    parser.add_argument('--ablation', type=str, default='full',
                        help=f'Ablation config. Choices: {ALL_ABLATIONS}, or "all" for full ablation study')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("\nWARNING: --device cuda requested but CUDA is not available.")
        print("  Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print("  Falling back to CPU.\n")
        device = 'cpu'
    else:
        device = args.device
    print(f"\nDevice: {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Determine which models to run
    if args.model == 'all':
        models_to_run = ALL_MODELS
    else:
        models_to_run = [args.model]

    # Determine which ablations to run
    if args.ablation == 'all':
        ablations_to_run = ALL_ABLATIONS
    else:
        ablations_to_run = [args.ablation]

    # Run benchmarks
    all_summaries = []
    for model_name in models_to_run:
        for ablation in ablations_to_run:
            # Only run ablations for riskmagnn variants
            if ablation != 'full' and model_name not in ('riskmagnn', 'riskmagnn_large'):
                continue
            try:
                summary = run_benchmark(model_name, args.seeds, args.split, device, ablation)
                all_summaries.append(summary)
            except Exception as e:
                print(f"\n  ERROR running {model_name} ({ablation}): {e}")
                import traceback
                traceback.print_exc()

    # Print comparison table
    if len(all_summaries) > 1:
        print_comparison_table(all_summaries)

    # Save combined results
    if all_summaries:
        combined_path = f'results/benchmark/combined_{args.split}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(combined_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
