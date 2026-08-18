"""
Chronological resplit of HBTBD dataset:
  Train: timesteps 1-40  (includes M2 edges from ts 35-40)
  Val:   timesteps 41-45
  Test:  timesteps 46-49

Creates data in data/hbtbd_chrono/train/, val/, test/
"""

import os
import numpy as np
from collections import defaultdict


def load_adjlist(filepath):
    """Load adjlist as dict: node_id -> [neighbor_ids]"""
    adj = defaultdict(list)
    if not os.path.exists(filepath):
        return adj
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                src = int(parts[0])
                if len(parts) > 1:
                    adj[src] = [int(x) for x in parts[1:]]
                else:
                    adj[src] = []
    return adj


def save_adjlist(adj_dict, filepath, num_nodes):
    """Save adjlist: one line per node (even if empty)"""
    with open(filepath, 'w') as f:
        for i in range(num_nodes):
            neighbors = adj_dict.get(i, [])
            if neighbors:
                f.write(f"{i} " + " ".join(str(n) for n in neighbors) + "\n")
            else:
                f.write(f"{i}\n")


def main():
    base = r"c:\Users\nandi\Desktop\GNN_AML\data\hbtbd\HBTBD\data"
    train_path = os.path.join(base, "train")
    test_path = os.path.join(base, "test")

    # Load features, labels, timesteps from both original splits
    train_features = np.load(os.path.join(train_path, "features0.npy"))
    test_features = np.load(os.path.join(test_path, "features0.npy"))
    train_labels = np.load(os.path.join(train_path, "labels.npy"))
    test_labels = np.load(os.path.join(test_path, "labels.npy"))
    train_ts = np.load(os.path.join(train_path, "timesteps.npy"))
    test_ts = np.load(os.path.join(test_path, "timesteps.npy"))

    n_train_orig = len(train_labels)
    n_test_orig = len(test_labels)
    n_total = n_train_orig + n_test_orig

    # Combine everything (test nodes get offset by n_train_orig)
    all_features = np.concatenate([train_features, test_features], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    all_ts = np.concatenate([train_ts, test_ts], axis=0)

    print(f"Combined: {n_total} nodes, timesteps {all_ts.min()}-{all_ts.max()}")
    print(f"Illicit rate: {all_labels.mean():.4f}")

    # Combine adjlists with offset
    combined_adjs = {}
    for mp in ['m1', 'm2', 'm3']:
        train_adj = load_adjlist(os.path.join(train_path, f"{mp}.adjlist"))
        test_adj = load_adjlist(os.path.join(test_path, f"{mp}.adjlist"))

        combined = defaultdict(list)
        for src, neighbors in train_adj.items():
            if src < n_train_orig:
                combined[src] = [n for n in neighbors if n < n_train_orig]
        for src, neighbors in test_adj.items():
            new_src = src + n_train_orig
            if new_src < n_total:
                combined[new_src] = [n + n_train_orig for n in neighbors if (n + n_train_orig) < n_total]

        total_edges = sum(len(v) for v in combined.values())
        combined_adjs[mp] = combined
        print(f"  {mp}: {total_edges} combined edges")

    # Chronological split
    train_mask = all_ts <= 40
    val_mask = (all_ts >= 41) & (all_ts <= 45)
    test_mask = all_ts >= 46

    splits = {
        "train": np.where(train_mask)[0],
        "val": np.where(val_mask)[0],
        "test": np.where(test_mask)[0],
    }

    for name, idx in splits.items():
        labels = all_labels[idx]
        ts = all_ts[idx]
        ill = labels.sum()
        print(f"\n{name}: {len(idx)} nodes, ts {ts.min()}-{ts.max()}, "
              f"{int(ill)} illicit ({ill/len(idx)*100:.1f}%)")

    # Save each split
    out_base = r"c:\Users\nandi\Desktop\GNN_AML\data\hbtbd_chrono"
    for split_name, split_idx in splits.items():
        out_path = os.path.join(out_base, split_name)
        os.makedirs(out_path, exist_ok=True)

        n_split = len(split_idx)
        old_to_new = {old: new for new, old in enumerate(split_idx)}
        split_set = set(split_idx.tolist())

        # Save features, labels, timesteps, node_types
        np.save(os.path.join(out_path, "features0.npy"), all_features[split_idx])
        np.save(os.path.join(out_path, "labels.npy"), all_labels[split_idx])
        np.save(os.path.join(out_path, "timesteps.npy"), all_ts[split_idx])
        np.save(os.path.join(out_path, "node_types.npy"),
                np.zeros(n_split, dtype=np.int64))

        # Remap and save adjlists
        for mp in ['m1', 'm2', 'm3']:
            new_adj = defaultdict(list)
            edge_count = 0
            for old_src, old_neighbors in combined_adjs[mp].items():
                if old_src in split_set:
                    new_src = old_to_new[old_src]
                    new_neighbors = []
                    for old_dst in old_neighbors:
                        if old_dst in split_set:
                            new_neighbors.append(old_to_new[old_dst])
                    new_adj[new_src] = new_neighbors
                    edge_count += len(new_neighbors)

            save_adjlist(new_adj, os.path.join(out_path, f"{mp}.adjlist"), n_split)
            print(f"  {split_name} {mp}: {edge_count} edges")

    print(f"\nDone! Data saved to: {out_base}")


if __name__ == "__main__":
    main()
