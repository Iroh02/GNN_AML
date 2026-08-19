# GNN-Based Anti-Money Laundering Detection

**Graph Neural Networks for Bitcoin Transaction Classification**

[![PR-AUC](https://img.shields.io/badge/PR--AUC-65.6%25%20%C2%B1%201.8-success)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.14-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)](https://pytorch.org)

Code for the AIxB 2026 paper *"A Graph Neural Network for Explainable
Anti-Money Laundering in Blockchain Transactions."*
**→ [REPRODUCIBILITY.md](REPRODUCIBILITY.md) has the exact commands, runtimes
and statistical protocol behind every number in the paper.**

---

## Quick Start

```bash
pip install -r requirements.txt
```

Reproduce the paper's main table (5 seeds, ~8 min on GPU):

```bash
python run_multiseed_all.py
```

Then the significance tests:

```bash
python analyze_significance.py
```

---

## Project Overview

This project develops Graph Neural Network models to detect illicit Bitcoin transactions (money laundering, fraud, ransomware) using two datasets:

- **Elliptic Dataset**: 203,769 transactions with temporal structure
- **HBTBD Dataset**: 46,045 transactions with heterogeneous wallet address nodes

### Headline Results

HBTBD figures are **mean ± sample std over 10 seeds** (five original + five
pre-registered extensions); Elliptic uses 5 seeds. One unified protocol;
significance tests are paired across seeds.

| Approach | Dataset | Test PR-AUC | Test ROC-AUC |
|----------|---------|-------------|--------------|
| Logistic Regression (feature-only) | Elliptic | 16.21 ± 0.00 | 82.44 ± 0.00 |
| **Deep MLP (no graph)** | Elliptic | 53.32 ± 3.67 | 81.87 ± 0.23 |
| Static GraphSAGE | Elliptic | 53.77 ± 0.98 | 81.75 ± 0.38 |
| Temporal GNN (timestep input) | Elliptic | 52.14 ± 1.79 | 81.59 ± 0.98 |
| **Deep MLP (no graph)** | HBTBD | 64.70 ± 2.24 | 88.65 ± 0.77 |
| SimpleHeteroGNN | HBTBD | 64.80 ± 1.22 | 87.62 ± 1.20 |
| RiskMAGNN (d=128, L=2) | HBTBD | 63.50 ± 6.78 | 89.95 ± 1.04 |
| RiskMAGNN (d=192, L=3) | HBTBD | 66.83 ± 4.04 | 90.15 ± 1.38 |
| **RiskMAGNN d=192, 10-seed ensemble** | **HBTBD** | **71.23** | **91.17** |

What survives seed control:

- **The headline "graph structure" effect is mostly model class.** A deep
  feature-only MLP recovers +37.11 of the 37.56 pp separating logistic
  regression from GraphSAGE; GraphSAGE leads the MLP by only +0.45 pp
  (*p* = 0.83). Both benchmarks' features already contain 1-hop/2-hop
  neighbourhood aggregates, so "feature-only" is not graph-blind.
- **Message passing pays off only with domain structure.** RiskMAGNN beats
  the MLP on ROC-AUC (+1.30 at d=128, *p* = 0.006; +1.50 at d=192,
  *p* = 0.033) and SimpleHeteroGNN by +2.33/+2.53 (*p* = 0.002), while naive
  metapath aggregation ranks *worse* than the MLP (−1.02, *p* = 0.034).
  PR-AUC differences among all single models are within seed noise.
- **Seed ensembling**: +4–5 PR-AUC points for every architecture — the
  cheapest reliable improvement on this benchmark.
- **The two proposed components jointly**: +2.72 PR-AUC points, *p* = 0.057
  at n=10 — suggestive, not confirmed. Training can collapse on single seeds
  (three of four ablation variants had ≥1 collapse), which is why single-seed
  results here should not be trusted.
- **The HBTBD val–test gap is temporal shift, not just the missing M2
  metapath**: the MLP uses no metapaths yet shows a 34.7 pp gap.

### Dataset finding: HBTBD ships an empty M2 training adjacency list

`m2.adjlist` for the training split is 0 bytes, but the release's own
instance file `idx01.pickle` contains **602,850 M2 training instances**
(382,342 with distinct endpoints) spanning training time steps 1–34. For M1
the two views match exactly (3,305,618 vs 3,305,618). Any pipeline built on
the adjacency lists therefore trains without the layering metapath while the
data sits unused in the same release. Reproduce with
`python build_address_features.py`.

The instance files also carry the intermediate **address node ids**, and
`features1-3.npy` hold 8 features per address node. The models in the paper
use none of this — they read only `features0.npy` and the adjacency lists.

> ⚠️ **Superseded numbers.** Earlier versions of this README and the submitted
> manuscript reported 69.02% PR-AUC for RiskMAGNN and 60.4% for
> SimpleHeteroGNN. Those were single-seed results measured under
> non-comparable protocols and **do not reproduce**. See
> [REPRODUCIBILITY.md §5](REPRODUCIBILITY.md#5-corrections-relative-to-the-submitted-version)
> for what changed and why. Files such as `results/riskmagnn_results.csv` are
> retained only as historical artefacts.

---

## Repository Structure

```
GNN_AML/
├── data/
│   ├── raw/                    # Elliptic dataset
│   └── hbtbd/                  # HBTBD heterogeneous dataset
├── src/
│   └── models/
│       ├── temporal_gnn.py     # Temporal GNN (Elliptic: 55.5%)
│       ├── improved_temporal_gnn.py  # Multi-scale temporal
│       ├── gnn_autoencoder.py  # Autoencoder approach
│       └── magnn.py            # Heterogeneous GNN (HBTBD: 60.4%)
├── run_hbtbd.py                # Train heterogeneous model (BEST)
├── run_improved_gnn.py         # Train temporal model
├── run_autoencoder.py          # Train autoencoder
├── results/                    # Experiment results
├── PROJECT_DOCUMENTATION.md    # Complete technical documentation
├── RESEARCH_PRESENTATION.md    # Academic presentation format
└── README.md                   # This file
```

---

## Models Implemented

### 1. Homogeneous Models (Elliptic Dataset)

| Model | Description | Test PR-AUC |
|-------|-------------|-------------|
| Logistic Regression | Non-graph baseline | 16.8% |
| Static GraphSAGE | Basic GNN | 48.7% |
| **Absolute Temporal GNN** | **Best homogeneous** | **55.5%** |
| Relative Temporal GNN | Time-aware, no leakage | 50.5% |
| Improved Temporal GNN | Multi-scale + domain features | 51.7% |
| GNN Autoencoder | Reconstruction-based | 47.5% |
| Enhanced GAT | Attention + SSL | 44.6% |

### 2. Heterogeneous Models (HBTBD Dataset)

| Model | Description | Test PR-AUC |
|-------|-------------|-------------|
| **SimpleHeteroGNN** | **Metapath aggregation** | **60.4%** ✓ |
| SimpleHeteroGNN (Large) | 3-layer, 192-dim | 53.6% |

---

## Key Features

### Heterogeneous Graph Structure (HBTBD)

The winning approach uses **heterogeneous graphs** with 4 node types:

```
Node Types:
  - Type 0: Transaction nodes (29,699 nodes, 165 features)
  - Type 1: Input Address nodes (68,560 nodes)
  - Type 2: Output Address nodes (113,121 nodes)
  - Type 3: Change Address nodes (16,796 nodes)

Metapaths:
  - M1: Tx → Input_Addr → Tx (423K edges)
  - M2: Tx → Output_Addr → Tx (~100 edges)
  - M3: Tx → Change_Addr → Tx (55K edges)
```

### Why It Works

Money launderers reuse wallets across multiple transactions. The heterogeneous graph reveals these connections:

```
Original Elliptic:
  Tx_A → Tx_B (direct transaction flow)

HBTBD Heterogeneous:
  Tx_A ←→ Wallet_X ←→ Tx_B (shared wallet)
  Tx_C ←→ Wallet_X ←→ Tx_D (same wallet reused)
```

If Tx_A is illicit and shares a wallet with Tx_B, C, D, then B, C, D are likely illicit too.

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.0+

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/GNN_AML.git
cd GNN_AML

# Install dependencies
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse
pip install pandas numpy scikit-learn scipy

# Download Elliptic dataset (if not included)
# Place in data/raw/

# Download HBTBD dataset
kaggle datasets download -d songjialin/hbtbd-for-aml -p data/hbtbd --unzip
```

---

## Usage

### Train Best Model (Heterogeneous GNN on HBTBD)

```bash
python run_hbtbd.py
```

**Expected Output:**
```
======================================================================
  HBTBD HETEROGENEOUS GRAPH EXPERIMENT
======================================================================

Training SimpleHeteroGNN...
  Parameters: 358,850
  Epoch  20: Loss=0.0167, Val PR-AUC=0.9144, F1=0.8068
  Epoch 100: Loss=0.0053, Val PR-AUC=0.9821, F1=0.9033
  Early stopping at epoch 194

  Final Results:
    Test PR-AUC:  0.6040  ✓
    Test ROC-AUC: 0.8699
    Test F1:      0.5900
    Precision:    0.6434
    Recall:       0.5448

  SUCCESS! Improved by +7.4 percentage points!
```

### Train Temporal GNN (Elliptic)

```bash
python run_improved_gnn.py
```

### Compare All Models

```bash
# View results
cat results/hbtbd_results.csv
cat results/improved_gnn_results.csv
```

---

## Results

### Performance Summary

| Metric | Baseline | Best Homogeneous | Best Heterogeneous | Target |
|--------|----------|------------------|-------------------|--------|
| **PR-AUC** | 16.8% | 55.5% | **60.4%** | 60-70% ✓ |
| **ROC-AUC** | 75.2% | 85.6% | **87.0%** | — |
| **F1 Score** | ~20% | ~52% | **59.0%** | — |
| **Precision** | — | — | **64.3%** | — |
| **Recall** | — | — | **54.5%** | — |

### What We Learned

✓ **What Works:**
1. Graph structure essential (+31.9 points)
2. Temporal encoding helps (+6.8 points)
3. **Heterogeneous graphs best** (+4.9 points)
4. Simple models generalize better
5. Metapaths capture wallet-sharing patterns

✗ **What Doesn't Work:**
1. Domain features add noise (-3.8 points)
2. Autoencoders fail on non-anomalous illicit transactions (-8.0 points)
3. Semi-supervised learning hurts with distribution shift (-3.0 points)
4. Attention mechanisms overfit (-10.9 points)
5. Larger models don't help (-6.8 points)

---

## Documentation

- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**: Complete technical documentation with architecture details, implementation notes, and reproducibility guide
- **[RESEARCH_PRESENTATION.md](RESEARCH_PRESENTATION.md)**: Academic presentation format with experimental results, analysis, and future work

---

## Datasets

### 1. Elliptic Dataset

**Source:** [Elliptic Co. & UCL (2019)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

```
Nodes:      203,769 Bitcoin transactions
Edges:      234,355 directed payment flows
Features:   166 per node
Timesteps:  49 time periods
Labels:     2% illicit, 21% licit, 77% unknown
```

### 2. HBTBD Dataset

**Source:** [Song & Gu (2023)](https://www.kaggle.com/datasets/songjialin/hbtbd-for-aml)

```
Transaction Nodes: 46,045
Address Nodes:     319,311 (3 types)
Metapath Edges:    478,731
Labels:            11.7% illicit (train), 6.6% (test)
```

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{gnn_aml_2026,
  title={Graph Neural Networks for Anti-Money Laundering Detection},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/your-username/GNN_AML}}
}
```

### Key References

1. Weber et al. (2019). "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"
2. Song & Gu (2023). "HBTBD: A Heterogeneous Bitcoin Transaction Behavior Dataset for Anti-Money Laundering"
3. Fu et al. (2020). "MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding"

---

## License

MIT License - see LICENSE file for details

---

## Contact

For questions or collaboration:
- Email: your.email@university.edu
- GitHub Issues: [Open an issue](https://github.com/your-username/GNN_AML/issues)

---

## Acknowledgments

- **Elliptic Co.** for providing the Bitcoin transaction dataset
- **Song & Gu (2023)** for the HBTBD heterogeneous dataset
- **PyTorch Geometric team** for excellent GNN library

---

**Status:** ✓ **Target Achieved (60.4% PR-AUC)**

*Last updated: January 22, 2026*
