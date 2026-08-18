"""
Resplit HBTBD dataset: combine train+test, then re-split 80/20
ensuring M2 edges appear in both splits.

Creates new data in data/hbtbd_resplit/train/ and data/hbtbd_resplit/test/
"""

import os
import numpy as np
import pickle
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

    # Load features and labels from both splits
    train_features = np.load(os.path.join(train_path, "features0.npy"))
    test_features = np.load(os.path.join(test_path, "features0.npy"))
    train_labels = np.load(os.path.join(train_path, "labels.npy"))
    test_labels = np.load(os.path.join(test_path, "labels.npy"))

    n_train = len(train_labels)
    n_test = len(test_labels)
    n_total = n_train + n_test

    print(f"Original: {n_train} train, {n_test} test, {n_total} total")
    print(f"Train labels: {np.unique(train_labels, return_counts=True)}")
    print(f"Test labels: {np.unique(test_labels, return_counts=True)}")

    # Combine features and labels
    all_features = np.concatenate([train_features, test_features], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    # Load and combine adjlists (remap test node IDs by adding n_train offset)
    print("\nLoading adjacency lists...")
    for mp in ['m1', 'm2', 'm3']:
        train_adj = load_adjlist(os.path.join(train_path, f"{mp}.adjlist"))
        test_adj = load_adjlist(os.path.join(test_path, f"{mp}.adjlist"))

        train_edges = sum(len(v) for v in train_adj.values())
        test_edges = sum(len(v) for v in test_adj.values())
        print(f"  {mp}: {train_edges} train edges, {test_edges} test edges")

    # Combine adjlists with offset
    combined_adjs = {}
    for mp in ['m1', 'm2', 'm3']:
        train_adj = load_adjlist(os.path.join(train_path, f"{mp}.adjlist"))
        test_adj = load_adjlist(os.path.join(test_path, f"{mp}.adjlist"))

        combined = defaultdict(list)
        # Train edges: keep as-is
        for src, neighbors in train_adj.items():
            if src < n_train:
                combined[src] = [n for n in neighbors if n < n_train]
        # Test edges: remap by adding n_train
        for src, neighbors in test_adj.items():
            new_src = src + n_train
            if new_src < n_total:
                combined[new_src] = [n + n_train for n in neighbors if (n + n_train) < n_total]

        combined_adjs[mp] = combined

    # Stratified random split: 80/20
    np.random.seed(42)

    illicit_idx = np.where(all_labels == 1)[0]
    licit_idx = np.where(all_labels == 0)[0]

    np.random.shuffle(illicit_idx)
    np.random.shuffle(licit_idx)

    n_illicit_train = int(0.8 * len(illicit_idx))
    n_licit_train = int(0.8 * len(licit_idx))

    new_train_idx = np.concatenate([
        illicit_idx[:n_illicit_train],
        licit_idx[:n_licit_train]
    ])
    new_test_idx = np.concatenate([
        illicit_idx[n_illicit_train:],
        licit_idx[n_licit_train:]
    ])

    np.random.shuffle(new_train_idx)
    np.random.shuffle(new_test_idx)

    print(f"\nNew split: {len(new_train_idx)} train, {len(new_test_idx)} test")
    print(f"New train illicit rate: {all_labels[new_train_idx].mean():.4f}")
    print(f"New test illicit rate: {all_labels[new_test_idx].mean():.4f}")

    # Create index mapping: old combined index -> new local index
    train_old_to_new = {old: new for new, old in enumerate(new_train_idx)}
    test_old_to_new = {old: new for new, old in enumerate(new_test_idx)}
    train_set = set(new_train_idx)
    test_set = set(new_test_idx)

    # Save new splits
    out_base = r"c:\Users\nandi\Desktop\GNN_AML\data\hbtbd_resplit"
    for split_name, split_idx, old_to_new, split_set in [
        ("train", new_train_idx, train_old_to_new, train_set),
        ("test", new_test_idx, test_old_to_new, test_set)
    ]:
        out_path = os.path.join(out_base, split_name)
        os.makedirs(out_path, exist_ok=True)

        # Save features and labels
        np.save(os.path.join(out_path, "features0.npy"), all_features[split_idx])
        np.save(os.path.join(out_path, "labels.npy"), all_labels[split_idx])

        # Save node_types (all tx = type 0)
        np.save(os.path.join(out_path, "node_types.npy"),
                np.zeros(len(split_idx), dtype=np.int64))

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

            save_adjlist(new_adj, os.path.join(out_path, f"{mp}.adjlist"), len(split_idx))
            print(f"  {split_name} {mp}: {edge_count} edges")

    print("\nDone! New data saved to:", out_base)

if __name__ == "__main__":
    main()
