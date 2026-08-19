"""
Build Per-Node Address-Feature Aggregates from HBTBD Metapath Instances
=======================================================================

HBTBD ships the metapath instances (idx00/01/02.pickle) as triples
(Tx_v, Address, Tx_u) together with 8-dimensional features for every address
node (features1/2/3.npy). The Tx-Tx adjacency lists our pipeline previously
consumed are those triples with the middle column removed, so the address
identity -- and hence MAGNN's intra-metapath instance encoding -- was being
discarded.

This script recovers it. For each transaction node v and metapath type m it
computes

    T_v^(m) = mean over v's metapath-m instances of the mediating address's
              8-dimensional feature vector

which is exactly the address term needed by a mean-pooled MAGNN instance
encoder. Because W and mean-pooling are linear and the relation vectors are
constant, mean-pooling the instance encoder over v's instances equals

    W [ h_v + T_v (*) r1 ; T_v ; N_v + T_v (*) r2 ]

with N_v the neighbour-transaction mean already produced by the sparse
adjacency product. So this per-node aggregate is sufficient to implement the
encoder faithfully, without materialising millions of instance embeddings.

Neighbour truncation matches the adjacency loader (first 50 instances per
node) so the two views stay consistent.

Outputs (per split): address_agg.npy of shape (num_tx, 3, 8), plus a mask
address_mask.npy of shape (num_tx, 3) marking nodes with at least one
instance for that metapath.
"""

import os
import pickle
import numpy as np

MAX_NEIGHBORS = 50  # must match load_adjlist() in run_riskmagnn.py
IDX_FILES = ['idx00.pickle', 'idx01.pickle', 'idx02.pickle']  # M1, M2, M3


def type_offsets(node_types):
    """Map node type -> (start_id, features_file_index)."""
    offsets, start = {}, 0
    for t in sorted(np.unique(node_types)):
        n = int((node_types == t).sum())
        offsets[int(t)] = (start, n)
        start += n
    return offsets


def build_split(data_path: str, verbose: bool = True):
    node_types = np.load(os.path.join(data_path, 'node_types.npy'))
    offsets = type_offsets(node_types)
    num_tx = int((node_types == 0).sum())

    # Address features live in features1/2/3.npy for node types 1/2/3.
    addr_feats = {}
    for t in [1, 2, 3]:
        p = os.path.join(data_path, f'features{t}.npy')
        addr_feats[t] = np.load(p) if os.path.exists(p) else None

    feat_dim = next(f.shape[1] for f in addr_feats.values() if f is not None)
    agg = np.zeros((num_tx, len(IDX_FILES), feat_dim), dtype=np.float32)
    mask = np.zeros((num_tx, len(IDX_FILES)), dtype=np.float32)

    for m, fname in enumerate(IDX_FILES):
        path = os.path.join(data_path, fname)
        if not os.path.exists(path):
            if verbose:
                print(f"  {fname}: missing -> metapath {m+1} left empty")
            continue

        with open(path, 'rb') as fh:
            idx = pickle.load(fh)

        n_inst, types_seen = 0, {}
        for v, arr in idx.items():
            arr = np.asarray(arr)
            if arr.size == 0:
                continue
            arr = arr.reshape(-1, 3)[:MAX_NEIGHBORS]
            v = int(v)
            if v >= num_tx:
                continue

            mids = arr[:, 1].astype(np.int64)
            vecs = np.zeros((len(mids), feat_dim), dtype=np.float32)
            ok = np.zeros(len(mids), dtype=bool)
            for t in [1, 2, 3]:
                if addr_feats[t] is None:
                    continue
                start, count = offsets[t]
                sel = (mids >= start) & (mids < start + count)
                if sel.any():
                    vecs[sel] = addr_feats[t][mids[sel] - start]
                    ok |= sel
                    types_seen[t] = types_seen.get(t, 0) + int(sel.sum())

            if ok.any():
                agg[v, m] = vecs[ok].mean(axis=0)
                mask[v, m] = 1.0
                n_inst += int(ok.sum())

        if verbose:
            covered = int(mask[:, m].sum())
            print(f"  {fname}: {n_inst:,} instances | {covered:,}/{num_tx:,} "
                  f"nodes covered | mediating node types {types_seen}")

    return agg, mask


def main():
    for split in ['train', 'test']:
        for base in ['data/hbtbd/HBTBD/data', 'data/hbtbd_resplit']:
            path = os.path.join(base, split)
            if not os.path.isdir(path):
                continue
            if not os.path.exists(os.path.join(path, 'idx00.pickle')):
                print(f"{path}: no idx files, skipping")
                continue
            print(f"\n{path}")
            agg, mask = build_split(path)
            np.save(os.path.join(path, 'address_agg.npy'), agg)
            np.save(os.path.join(path, 'address_mask.npy'), mask)
            print(f"  saved address_agg.npy {agg.shape}, "
                  f"address_mask.npy {mask.shape}")


if __name__ == "__main__":
    main()
