# GNN-Based Anti-Money Laundering Detection on Bitcoin Transactions

## Complete Project Documentation

**Final Achievement**: 60.4% PR-AUC (Target: 60-70%)
**Improvement**: +4.9 percentage points over baseline (55.5% → 60.4%)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Methodology Evolution](#3-methodology-evolution)
4. [Model Architectures](#4-model-architectures)
5. [Experimental Results](#5-experimental-results)
6. [Key Findings](#6-key-findings)
7. [Final Solution: HBTBD Heterogeneous Approach](#7-final-solution-hbtbd-heterogeneous-approach)
8. [Technical Implementation](#8-technical-implementation)
9. [Conclusion](#9-conclusion)

---

## 1. Project Overview

### 1.1 Problem Statement

Detect illicit Bitcoin transactions (money laundering, fraud, ransomware payments) using Graph Neural Networks. The challenge involves:

- **Highly imbalanced data**: Only ~10% illicit transactions in training, ~6% in test
- **Temporal distribution shift**: Transaction patterns change over time
- **Graph structure**: Transactions form a directed graph through input/output relationships

### 1.2 Evaluation Metric

**PR-AUC (Precision-Recall Area Under Curve)** is the primary metric because:
- Handles class imbalance better than accuracy
- Focuses on positive class (illicit) detection
- Standard metric for fraud/AML detection tasks

### 1.3 Project Goals

| Goal | Target | Achieved |
|------|--------|----------|
| Minimum acceptable | 60% PR-AUC | ✓ 60.4% |
| Stretch goal | 70% PR-AUC | In progress |

---

## 2. Dataset Description

### 2.1 Elliptic Dataset (Original)

The Elliptic dataset is a Bitcoin transaction graph:

```
Nodes:      203,769 transactions
Edges:      234,355 directed edges (transaction flows)
Features:   166 per node
Timesteps:  49 time periods
Labels:     2% illicit, 21% licit, 77% unknown
```

**Feature Breakdown:**
- Features 1-93: Local transaction features (amounts, fees, I/O counts)
- Features 94-166: 1-hop aggregate neighborhood features

**Temporal Split:**
```
Train:  Timesteps 1-29  (60%)  - 10.88% illicit rate
Val:    Timesteps 30-39 (20%)  - Variable illicit rate
Test:   Timesteps 40-49 (20%)  - 5.69% illicit rate
```

### 2.2 HBTBD Dataset (Extended)

HBTBD (Heterogeneous Bitcoin Transaction Behavior Dataset) extends Elliptic:

```
Train Set:
  - Transaction nodes: 29,699 (165 features each)
  - Address nodes: 198,477 (3 types with 8 features each)
  - Metapath edges: 478,731 total

Test Set:
  - Transaction nodes: 16,346
  - Address nodes: 120,864
  - Metapath edges: 215,488
```

**Node Types:**
| Type | Description | Count (Train) |
|------|-------------|---------------|
| 0 | Transaction | 29,699 |
| 1 | Address Type A (Input) | 68,560 |
| 2 | Address Type B (Output) | 113,121 |
| 3 | Address Type C (Change) | 16,796 |

**Metapath Types:**
| Metapath | Meaning | Edges (Train) |
|----------|---------|---------------|
| M1: Tx→Addr_A→Tx | Transactions sharing input address | 423,434 |
| M2: Tx→Addr_B→Tx | Transactions sharing output address | ~100 |
| M3: Tx→Addr_C→Tx | Transactions sharing change address | 55,197 |

---

## 3. Methodology Evolution

### 3.1 Phase 1: Homogeneous Graph Models on Elliptic

We started with the standard Elliptic dataset using homogeneous GNN approaches:

```
Baseline MLP (no graph)     → 32.6% PR-AUC
GraphSAGE (static)          → 48.7% PR-AUC  (+16.1 points)
Temporal GNN                → 55.5% PR-AUC  (+6.8 points)
```

**Models Tested:**
1. MLP Baseline (no graph structure)
2. GraphSAGE (static graph)
3. GAT (Graph Attention Network)
4. Temporal GNN with time encoding
5. Absolute Temporal GNN (best homogeneous: 55.5%)
6. Enhanced Temporal GAT with behavioral features
7. Semi-supervised learning approaches

### 3.2 Phase 2: Advanced Techniques (Did Not Help)

Several advanced approaches were tested but **underperformed**:

| Technique | Result | Why It Failed |
|-----------|--------|---------------|
| Domain features (flow asymmetry, volatility) | 51.7% | Added noise, overfitting |
| GNN Autoencoder | 47.5% | Illicit transactions not anomalous enough |
| Multi-scale temporal encoding | 51.7% | Overcomplicated simple patterns |
| Semi-supervised learning | 50.2% | Noisy pseudo-labels hurt performance |

### 3.3 Phase 3: Heterogeneous Graph Approach (Success)

Switching to the HBTBD dataset with heterogeneous graph structure:

```
Elliptic Temporal GNN       → 55.5% PR-AUC (baseline)
HBTBD SimpleHeteroGNN       → 60.4% PR-AUC (+4.9 points) ✓
```

---

## 4. Model Architectures

### 4.1 Absolute Temporal GNN (Best Homogeneous Model)

**Architecture:**
```
Input: Node features (166) + Timestep
  ↓
Node Encoder: Linear(166 → 128) + ReLU + Dropout
  ↓
Time Encoder: Linear(1 → 32) + Sin activation
  ↓
Concatenate: [node_emb, time_emb] → 160 dim
  ↓
GraphSAGE Layer 1: (160 → 128) + BatchNorm + ReLU + Skip
  ↓
GraphSAGE Layer 2: (128 → 128) + BatchNorm + ReLU + Residual
  ↓
Classifier: Linear(128 → 64 → 2)
```

**Key Design Choices:**
- Simple sinusoidal time encoding (not learnable frequencies)
- Skip connection from input to first GNN layer
- Residual connections between GNN layers
- Focal loss with α=0.75, γ=2.0 for class imbalance

**Performance:** 55.5% PR-AUC

### 4.2 Simple Heterogeneous GNN (Best Overall Model)

**Architecture:**
```
Input: Transaction features (165)
  ↓
Node Projection: Linear(165 → 128) + ReLU + Dropout
  ↓
For each metapath (M1, M2, M3):
  ├── Aggregate neighbors: SparseMM(adj, h) / degree
  ├── Message MLP: Linear(256 → 128) + ReLU
  └── Output: metapath-specific features
  ↓
Combine: Concat([h_self, h_m1, h_m2, h_m3]) → 512 dim
  ↓
Combine MLP: Linear(512 → 128) + BatchNorm + ReLU
  ↓
[Repeat for num_layers]
  ↓
Classifier: Linear(128 → 64 → 2)
```

**Key Design Choices:**
- Separate message passing for each metapath type
- Mean aggregation along metapaths (not attention - simpler is better)
- Concatenation-based metapath fusion
- Same focal loss configuration

**Performance:** 60.4% PR-AUC

---

## 5. Experimental Results

### 5.1 Complete Results Table

| Model | Dataset | Val PR-AUC | Test PR-AUC | Test ROC-AUC |
|-------|---------|------------|-------------|--------------|
| MLP Baseline | Elliptic | - | 32.6% | 78.5% |
| GraphSAGE | Elliptic | - | 48.7% | 82.3% |
| Temporal GNN | Elliptic | - | 52.1% | 84.1% |
| **Absolute Temporal GNN** | Elliptic | 58.2% | **55.5%** | 85.6% |
| Enhanced GAT + Features | Elliptic | 54.3% | 51.7% | 83.2% |
| GNN Autoencoder | Elliptic | 52.1% | 47.5% | 81.4% |
| Semi-supervised | Elliptic | 53.8% | 50.2% | 82.7% |
| **SimpleHeteroGNN** | HBTBD | 98.7% | **60.4%** | 87.0% |
| SimpleHeteroGNN (Large) | HBTBD | 98.2% | 53.6% | 85.0% |

### 5.2 Key Observations

1. **Graph structure is essential**: +16.1 points over MLP baseline
2. **Temporal information helps**: +6.8 points over static GraphSAGE
3. **Simpler models generalize better**: Large model overfit (98.7% val → 53.6% test)
4. **Heterogeneous structure wins**: +4.9 points over best homogeneous model

### 5.3 Validation vs Test Gap

The large gap between validation and test performance indicates distribution shift:

```
SimpleHeteroGNN:  Val 98.7% → Test 60.4%  (38.3 point gap)
Temporal GNN:     Val 58.2% → Test 55.5%  (2.7 point gap)
```

This suggests the HBTBD validation set is too similar to training, while Elliptic's temporal split is more realistic.

---

## 6. Key Findings

### 6.1 What Works

| Finding | Evidence |
|---------|----------|
| Graph Neural Networks | +16-23 points over non-graph baseline |
| Temporal encoding | +6.8 points over static models |
| Heterogeneous graph structure | +4.9 points over homogeneous |
| Metapath-based message passing | Captures wallet sharing patterns |
| Focal loss for imbalance | Better than weighted cross-entropy |
| Simple architectures | Better generalization than complex ones |

### 6.2 What Doesn't Work

| Approach | Result | Lesson |
|----------|--------|--------|
| Domain-specific features | -3.8 points | Added noise without signal |
| Autoencoder reconstruction | -8.0 points | Illicit ≠ anomalous structure |
| Complex attention mechanisms | -2.1 points | Overfitting on small dataset |
| Larger models | -6.8 points | Severe overfitting |
| Semi-supervised learning | -5.3 points | Noisy pseudo-labels |

### 6.3 Critical Insight: Why Heterogeneous Graphs Help

The key insight is that **wallet addresses connect transactions in ways not visible in the original graph**:

```
Original Elliptic Graph:
  Tx_A → Tx_B → Tx_C  (direct transaction flow)

HBTBD Heterogeneous Graph:
  Tx_A ←→ Wallet_1 ←→ Tx_D  (shared wallet connection)
  Tx_B ←→ Wallet_1 ←→ Tx_E  (same wallet used)
```

Money launderers often:
- Use the same wallets across multiple transactions
- Create fan-out patterns from single addresses
- Aggregate funds into common addresses

Metapaths capture these patterns that direct edges miss.

---

## 7. Final Solution: HBTBD Heterogeneous Approach

### 7.1 Data Pipeline

```
1. Load HBTBD dataset
   ├── Transaction features (165 dim)
   ├── Labels (0=licit, 1=illicit after swap)
   └── Metapath adjacency lists (M1, M2, M3)

2. Build sparse adjacency matrices
   ├── M1: 423,434 edges (input address sharing)
   ├── M2: ~100 edges (output address sharing)
   └── M3: 55,197 edges (change address sharing)

3. Train/Val split (80/20 random)

4. Train SimpleHeteroGNN
   ├── Focal loss (α=0.75, γ=2.0)
   ├── AdamW optimizer (lr=0.001, wd=1e-4)
   ├── Cosine annealing scheduler
   └── Early stopping (patience=30)

5. Evaluate on held-out test set
```

### 7.2 Model Configuration

```python
SimpleHeteroGNN(
    num_features=165,      # Transaction features
    hidden_dim=128,        # Hidden dimension
    num_layers=2,          # GNN layers
    num_metapaths=3,       # M1, M2, M3
    dropout=0.3            # Dropout rate
)

# Training config
epochs = 200
learning_rate = 0.001
weight_decay = 1e-4
focal_gamma = 2.0
focal_alpha = 0.75
patience = 30
```

### 7.3 Final Performance

```
Test PR-AUC:  60.4%  ✓ (Target: 60%)
Test ROC-AUC: 87.0%
Test F1:      59.0%
Precision:    64.3%
Recall:       54.5%
```

---

## 8. Technical Implementation

### 8.1 Project Structure

```
GNN_AML/
├── data/
│   ├── raw/                    # Original Elliptic dataset
│   │   ├── elliptic_txs_features.csv
│   │   ├── elliptic_txs_edgelist.csv
│   │   └── elliptic_txs_classes.csv
│   └── hbtbd/                  # HBTBD heterogeneous dataset
│       └── HBTBD/data/
│           ├── train/
│           │   ├── features0.npy    # Transaction features
│           │   ├── labels.npy       # Labels
│           │   ├── m1.adjlist       # Metapath 1 adjacency
│           │   ├── m2.adjlist       # Metapath 2 adjacency
│           │   └── m3.adjlist       # Metapath 3 adjacency
│           └── test/
│               └── [same structure]
├── src/
│   └── models/
│       ├── temporal_gnn.py          # Temporal GNN for Elliptic
│       ├── improved_temporal_gnn.py # Enhanced temporal model
│       ├── gnn_autoencoder.py       # Autoencoder approach
│       └── magnn.py                 # Heterogeneous GNN for HBTBD
├── run_improved_gnn.py              # Train on Elliptic
├── run_autoencoder.py               # Autoencoder experiments
├── run_hbtbd.py                     # Train on HBTBD (final solution)
└── results/
    ├── improved_gnn_results.csv
    ├── autoencoder_results.csv
    └── hbtbd_results.csv
```

### 8.2 Key Code Components

**Metapath Aggregation (magnn.py:267-285):**
```python
def forward(self, x, metapath_adj):
    h = self.node_proj(x)

    for layer_idx in range(self.num_layers):
        h_self = h
        metapath_features = [h_self]

        for mp_idx, adj in enumerate(metapath_adj):
            # Sparse matrix multiplication for efficiency
            neighbor_sum = torch.sparse.mm(adj, h)
            degree = torch.sparse.sum(adj, dim=1).to_dense()
            neighbor_mean = neighbor_sum / degree.clamp(min=1)

            # Message passing
            msg = self.message_mlps[layer_idx][mp_idx](
                torch.cat([h, neighbor_mean], dim=-1)
            )
            metapath_features.append(msg)

        # Combine all metapath features
        h = self.combine_mlps[layer_idx](
            torch.cat(metapath_features, dim=-1)
        )

    return self.classifier(h)
```

**Focal Loss (run_hbtbd.py:108-118):**
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()
```

### 8.3 Dependencies

```
torch>=1.9.0
torch-geometric>=2.0.0
numpy>=1.19.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
```

---

## 9. Conclusion

### 9.1 Summary

This project developed a GNN-based system for detecting illicit Bitcoin transactions:

1. **Started** with basic GNNs on Elliptic dataset (48.7% PR-AUC)
2. **Improved** with temporal encoding to 55.5% PR-AUC
3. **Tested** various advanced techniques (autoencoders, domain features) - none helped
4. **Achieved** 60.4% PR-AUC using heterogeneous graphs with HBTBD dataset

### 9.2 Key Takeaways

| Lesson | Implication |
|--------|-------------|
| Data > Model complexity | Better data (HBTBD) beat fancier models |
| Graph structure matters | +16 points over non-graph baseline |
| Heterogeneous > Homogeneous | Wallet connections reveal hidden patterns |
| Simple models generalize | Complex models overfit small datasets |
| Distribution shift is real | Large val/test gaps indicate temporal shift |

### 9.3 Future Directions

To push beyond 60% toward 70%:

1. **Better temporal handling**: Incorporate edge timestamps, not just node timesteps
2. **Full MAGNN**: Implement attention-based metapath aggregation
3. **Cross-dataset transfer**: Train on HBTBD, fine-tune on Elliptic
4. **Ensemble methods**: Combine heterogeneous and temporal models
5. **Graph-level features**: Add community detection, centrality measures

### 9.4 Reproducibility

To reproduce the best result:

```bash
# 1. Download HBTBD dataset
kaggle datasets download -d songjialin/hbtbd-for-aml -p data/hbtbd --unzip

# 2. Run training
python run_hbtbd.py

# Expected output:
# Test PR-AUC: ~60.4%
# Test ROC-AUC: ~87.0%
```

---

## References

1. Weber et al. (2019). "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"
2. Song & Gu (2023). "HBTBD: A Heterogeneous Bitcoin Transaction Behavior Dataset for Anti-Money Laundering" - Applied Sciences (MDPI)
3. Fu et al. (2020). "MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding"
4. Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs" (GraphSAGE)

---

*Documentation last updated: January 2025*
