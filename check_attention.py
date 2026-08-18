"""
Measure the learned per-metapath attention weights of the saved checkpoints.

Backs the camera-ready claim that the model down-weights M2 relative to its
forensic prior: the initial biases alone give M2 an attention share of 0.43,
while the learned mean M2 attention is ~0.29 on the official split (M2 absent
from training) and ~0.17 on the resplit (M2 present but sparse).

Requires: models/riskmagnn_large.pth, models/riskmagnn_resplit.pth, and both
datasets on disk. Runtime < 2 min on CPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models.riskmagnn import create_riskmagnn
from run_riskmagnn import load_hbtbd_data


def report(name, ckpt, train_path, test_path):
    # Scaler must be fitted on the split's own training data (as in training).
    _, _, _, _, scaler = load_hbtbd_data(train_path, normalize=True)
    test_feat, _, _, test_adj, _ = load_hbtbd_data(
        test_path, normalize=True, scaler=scaler)

    model = create_riskmagnn(num_features=test_feat.shape[1],
                             hidden_dim=192, num_layers=3)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        attn = model.get_attention_weights(test_feat, test_adj)

    print(f"\n{name}  ({ckpt})")
    for i, a in enumerate(attn):
        m = a.mean(dim=0)
        print(f"  layer {i}: M1={m[0]:.3f}  M2={m[1]:.3f}  M3={m[2]:.3f}")
    overall = torch.stack([a.mean(dim=0) for a in attn]).mean(dim=0)
    print(f"  mean over layers: M1={overall[0]:.3f}  "
          f"M2={overall[1]:.3f}  M3={overall[2]:.3f}")
    rb = [state[k] for k in state if 'risk_bias' in k]
    print("  raw risk_bias params:", [[f"{v:.3f}" for v in t] for t in rb])


if __name__ == "__main__":
    init = torch.softmax(torch.tensor([0.3, 0.6, 0.1]), dim=0)
    print("Attention share implied by the initial biases alone: "
          f"M1={init[0]:.3f}  M2={init[1]:.3f}  M3={init[2]:.3f}")

    report("OFFICIAL split (Large)", "models/riskmagnn_large.pth",
           "data/hbtbd/HBTBD/data/train/", "data/hbtbd/HBTBD/data/test/")
    report("RESPLIT (Large)", "models/riskmagnn_resplit.pth",
           "data/hbtbd_resplit/train/", "data/hbtbd_resplit/test/")
