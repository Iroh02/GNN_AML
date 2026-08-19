"""
Deconfounding the Resplit Diagnostic
====================================

The stratified resplit changes TWO things relative to the official split:
(a) M2 edges become available during training, and (b) train and test are
drawn from the same (mixed-timestep) feature distribution. The observed gap
collapse (29.7 pp -> 3.2 pp) could come from either. Two controls separate
them, both on the same resplit data and 5 seeds {0,1,7,42,123}:

  1. RiskMAGNN (Large) with the M2 adjacency ZEROED in train and test.
     If it still reaches ~94% test PR-AUC, the collapse was (b), not (a).

  2. A feature-only MLP (no metapaths at all).
     Bounds how much of resplit performance needs any graph input.

Protocol identical to run_resplit_seeds.py / run_multiseed_extended.py.

Output: results/resplit_attribution.json / .csv
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from src.models.riskmagnn import create_riskmagnn
from run_multiseed_all import EPOCHS, NO_EARLY_STOP, DEVICE, set_seed
from run_riskmagnn import train_model, evaluate
from run_resplit_seeds import prepare_resplit
from run_mlp_hbtbd import FeatureOnlyMLP, run_seed as mlp_run_seed_official

SEEDS = [0, 1, 7, 42, 123]


def empty_like(adj):
    n = adj.shape[0]
    return torch.sparse_coo_tensor(
        torch.zeros(2, 0, dtype=torch.long, device=adj.device),
        torch.zeros(0, device=adj.device), (n, n)).coalesce()


def run_riskmagnn_no_m2(data, seed):
    set_seed(seed)
    model = create_riskmagnn(
        num_features=data['train_features'].shape[1],
        hidden_dim=192, num_layers=3, num_metapaths=3, dropout=0.40,
    ).to(DEVICE)

    # Zero out M2 (index 1) everywhere: the model keeps its M2 pathway but it
    # receives no edges, exactly as on the official split.
    tr_adj = [data['train_adj'][0], empty_like(data['train_adj'][1]), data['train_adj'][2]]
    va_adj = [data['val_adj'][0], empty_like(data['val_adj'][1]), data['val_adj'][2]]
    te_adj = [data['test_adj'][0], empty_like(data['test_adj'][1]), data['test_adj'][2]]

    model, val_pr = train_model(
        model,
        data['train_features'], data['train_labels'], tr_adj,
        data['val_features'], data['val_labels'], va_adj,
        epochs=EPOCHS, patience=NO_EARLY_STOP,
        name=f"RiskMAGNN Large resplit NO-M2 seed={seed}",
    )
    m = evaluate(model, data['test_features'], data['test_labels'], te_adj)
    return {'variant': 'RiskMAGNN Large (resplit, M2 zeroed)', 'seed': seed,
            'val_pr_auc': float(val_pr), 'test_pr_auc': float(m['pr_auc']),
            'test_roc_auc': float(m['roc_auc']), 'test_f1': float(m['f1'])}


def run_mlp_resplit(data, seed):
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    from run_riskmagnn import FocalLoss

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
    return {'variant': 'MLP (resplit, no metapaths)', 'seed': seed,
            'val_pr_auc': float(best_val_pr),
            'test_pr_auc': float(average_precision_score(y, probs)),
            'test_roc_auc': float(roc_auc_score(y, probs)),
            'test_f1': float(f1_score(y, preds, zero_division=0))}


def main():
    print("RESPLIT ATTRIBUTION CONTROLS -- seeds", SEEDS)
    data = prepare_resplit()

    records = []
    for seed in SEEDS:
        r = run_riskmagnn_no_m2(data, seed)
        print(f"  NO-M2  seed {seed}: PR {r['test_pr_auc']:.4f} "
              f"(val {r['val_pr_auc']:.4f})")
        records.append(r)
    for seed in SEEDS:
        r = run_mlp_resplit(data, seed)
        print(f"  MLP    seed {seed}: PR {r['test_pr_auc']:.4f} "
              f"(val {r['val_pr_auc']:.4f})")
        records.append(r)

    print("\nSummary (test PR-AUC, mean +/- std):")
    for v in ['RiskMAGNN Large (resplit, M2 zeroed)', 'MLP (resplit, no metapaths)']:
        vals = np.array([r['test_pr_auc'] for r in records if r['variant'] == v]) * 100
        gaps = np.array([(r['val_pr_auc'] - r['test_pr_auc']) for r in records
                         if r['variant'] == v]) * 100
        print(f"  {v:<38} {vals.mean():.2f} +/- {vals.std(ddof=1):.2f} "
              f"(gap {gaps.mean():.2f} +/- {gaps.std(ddof=1):.2f} pp)")

    os.makedirs('results', exist_ok=True)
    with open('results/resplit_attribution.json', 'w') as f:
        json.dump({'seeds': SEEDS, 'per_run': records}, f, indent=2)
    import pandas as pd
    pd.DataFrame(records).to_csv('results/resplit_attribution.csv', index=False)
    print("\nSaved: results/resplit_attribution.json/.csv")


if __name__ == "__main__":
    main()
