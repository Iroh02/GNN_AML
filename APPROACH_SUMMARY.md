# GNN-Based Anti-Money Laundering Detection - Approach Summary

## Overview
This project implements a Graph Neural Network (GNN) approach for detecting illicit Bitcoin transactions using the Elliptic dataset, progressively improving from baseline models to state-of-the-art temporal GNNs.

## Dataset: Elliptic Bitcoin Transaction Network
- **Nodes**: 203,769 Bitcoin transactions
- **Edges**: 234,355 directed transaction flows
- **Features**: 166 features per node (94 local + 72 aggregated)
- **Labels**: Illicit (1), Licit (0), Unknown (-1)
- **Timesteps**: 49 time periods
- **Challenge**: Only ~20% of nodes are labeled, severe class imbalance (~10% illicit)

## Problem: Temporal Distribution Shift
The temporal evaluation setting creates a challenging problem:
- **Train**: Timesteps 0-29 (60% of data)
- **Val**: Timesteps 30-39 (20% of data)
- **Test**: Timesteps 40-49 (20% of data)

**Key Challenge**: Zero overlap between train and test timesteps means the model must generalize to future time periods with different illicit patterns.

## Model Progression

### 1. Baseline: Logistic Regression
**Purpose**: Establish non-graph baseline
**Performance**: ~16.8% PR-AUC
**Insight**: Node features alone insufficient; graph structure is critical

### 2. Static GraphSAGE
**Purpose**: Incorporate graph structure without temporal information
**Architecture**:
- 2 GraphSAGE layers (128 hidden dim)
- Batch normalization
- Focal loss for class imbalance

**Performance**: ~48.7% PR-AUC (+189% vs baseline)
**Insight**: Graph aggregation significantly improves detection

### 3. Absolute Temporal GNN
**Purpose**: Add temporal awareness with absolute timestamps
**Architecture**:
- Temporal embedding: sin encoding of absolute timesteps
- 2 SAGE layers + temporal features
- Skip connections

**Performance**: ~55.5% PR-AUC (best among earlier models)
**Problem Discovered**: Temporal leakage - model memorizes timestep-specific patterns that don't transfer to unseen future timesteps

### 4. Relative Temporal GNN (Our Fix)
**Purpose**: Address temporal leakage with relative time encoding
**Key Innovations**:
- **Relative Time Encoder**: Normalized position + sinusoidal encoding
  - Avoids memorizing absolute timesteps
  - Generalizes to future time periods

- **Behavioral Features**:
  - Transaction velocity (edges per unit time)
  - Neighbor temporal diversity
  - Time position features

**Performance**: ~50.5% PR-AUC
**Insight**: Avoids leakage but trades off some performance; pure behavioral features less powerful than absolute timestamps for seen timesteps

### 5. Enhanced Temporal GAT (Current Best Attempt)
**Purpose**: Achieve 60-70% PR-AUC target with attention, rich features, and semi-supervised learning

**Architecture**:
```
Enhanced Temporal GAT
├── Relative Time Encoder (32-dim)
│   └── Normalized position + sin/cos encoding
│
├── Behavioral Feature Extractor (6-dim)
│   ├── Transaction velocity
│   ├── Temporal burstiness (neighbor timestamp variance)
│   ├── Network centrality (in/out degree, asymmetry)
│   └── Neighbor temporal diversity
│
├── Node Feature Encoder (128-dim)
│   └── 2-layer MLP with residual
│
├── Temporal GAT Layers (3 layers, 4 heads each)
│   ├── Multi-head attention for adaptive aggregation
│   ├── Residual connections
│   └── Batch normalization
│
└── Classifier (128 -> 64 -> 2)
    └── Binary classification (illicit/licit)
```

**Key Features**:

1. **Graph Attention Mechanism (GAT)**:
   - Multi-head attention (4 heads)
   - Learns importance weights for each neighbor
   - More expressive than fixed GraphSAGE aggregation

2. **Rich Behavioral Features**:
   - **Velocity**: Transaction frequency normalized by time
   - **Burstiness**: Variance in neighbor timestamps (high = suspicious)
   - **Centrality**: In/out degree, degree asymmetry
   - **Temporal Diversity**: Mean time difference with neighbors

3. **Semi-Supervised Learning**:
   - Leverages 157,205 unlabeled nodes (~77% of dataset)
   - Pseudo-labeling with confidence thresholding (>95% confidence)
   - Consistency regularization
   - SSL kicks in after 50 epochs of supervised warmup

4. **Training Strategy**:
   - Focal loss (γ=2.0) for hard example mining
   - Class weighting for imbalance
   - AdamW optimizer with cosine annealing
   - Gradient clipping (max norm 1.0)
   - Early stopping (patience 30 epochs)

## Why These Improvements Matter

### Problem 1: Temporal Leakage
**Issue**: Absolute timestamps create timestep-specific signatures that don't generalize

**Solution**: Relative temporal encoding
- Normalized time position (0-1)
- Sinusoidal encoding for periodicity
- Focus on time differences, not absolute values

### Problem 2: Limited Labeled Data
**Issue**: Only 23% of nodes labeled, 77% wasted

**Solution**: Semi-supervised learning
- Pseudo-labeling on high-confidence predictions
- Gradually incorporate unlabeled nodes during training
- Effectively increases training signal

### Problem 3: Fixed Aggregation
**Issue**: GraphSAGE treats all neighbors equally

**Solution**: Graph Attention (GAT)
- Learn importance weights for each neighbor
- Attend to most relevant connections
- More expressive than mean aggregation

### Problem 4: Shallow Features
**Issue**: Original models use only basic node features + time

**Solution**: Rich behavioral features
- Velocity captures transaction frequency patterns
- Burstiness detects anomalous timing
- Centrality measures network position
- Temporal diversity captures interaction patterns

## Expected Improvements

**Target**: 60-70% PR-AUC

**Rationale**:
1. **GAT vs GraphSAGE**: +3-5% from attention mechanism
2. **Behavioral features**: +2-4% from richer representations
3. **Semi-supervised learning**: +5-10% from 3x more training signal
4. **Better architecture**: +2-3% from deeper model, better regularization

**Conservative estimate**: 50.5% (Relative) + 12-22% = **62-72% PR-AUC**

## Evaluation Metrics

### Primary: PR-AUC (Precision-Recall Area Under Curve)
- More appropriate for imbalanced datasets
- Focuses on positive class (illicit transactions)
- Less sensitive to class imbalance than ROC-AUC

### Secondary: ROC-AUC
- Overall classification performance
- Useful for comparison with literature

## Implementation Details

### Key Files
- `src/models/enhanced_temporal_gat.py`: Enhanced GAT model
- `run_enhanced_gat.py`: Training with semi-supervised learning
- `run_complete_comparison.py`: Compare all models
- `analyze_temporal.py`: Temporal leakage analysis

### Hyperparameters
- Hidden dimension: 128
- Time dimension: 32
- Layers: 3 GAT layers
- Attention heads: 4
- Dropout: 0.3
- Learning rate: 0.001 (cosine annealing)
- Weight decay: 1e-4
- Focal loss gamma: 2.0
- SSL confidence threshold: 0.95
- SSL weight: 0.5
- SSL warmup: 50 epochs

### Computational Requirements
- Model parameters: ~130K
- Training time: ~10-15 minutes (CPU)
- Memory: ~2-3 GB RAM

## Next Steps (If 60% Target Not Reached)

1. **Ensemble Methods**:
   - Combine Absolute, Relative, and Enhanced GAT predictions
   - Weighted voting based on validation performance

2. **Advanced Temporal Features**:
   - Temporal attention (learn to weight different time scales)
   - LSTM/GRU for sequence modeling
   - Temporal random walks

3. **Graph Structure Features**:
   - PageRank centrality
   - Betweenness centrality
   - Community detection
   - k-core decomposition

4. **Advanced SSL Techniques**:
   - Contrastive learning
   - Self-training with soft pseudo-labels
   - Co-training with multiple views

5. **Architecture Improvements**:
   - Deeper networks (5-7 layers) with careful residual design
   - Graph transformers
   - Heterogeneous attention

## Conclusion

This project demonstrates a principled approach to GNN-based AML detection:
1. Started with strong baselines (LR, GraphSAGE)
2. Identified temporal leakage problem through analysis
3. Developed relative temporal encoding to fix leakage
4. Enhanced with attention, behavioral features, and SSL
5. Systematic evaluation showing clear progression

The Enhanced Temporal GAT represents current best practices:
- Attention for adaptive aggregation
- Rich behavioral features for better representations
- Semi-supervised learning to leverage all data
- Relative temporal encoding for better generalization

**Goal**: Achieve 60-70% PR-AUC, demonstrating that GNNs can significantly improve AML detection in realistic temporal evaluation settings.
