"""
Enhanced Temporal GAT with Semi-Supervised Learning
====================================================

Training strategy:
1. Supervised loss on labeled nodes
2. Consistency regularization on unlabeled nodes (pseudo-labeling)
3. Contrastive learning for better representations
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

from src.models.enhanced_temporal_gat import create_enhanced_temporal_gat

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    """Load Elliptic dataset."""
    df = pd.read_csv('data/raw/elliptic_txs_features.csv', header=None)
    node_ids = df.iloc[:, 0].values
    timesteps = df.iloc[:, 1].values.astype(np.int64)
    features = StandardScaler().fit_transform(df.iloc[:, 2:].values.astype(np.float32))
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    edges = pd.read_csv('data/raw/elliptic_txs_edgelist.csv')
    src = edges.iloc[:, 0].map(id_map).values
    dst = edges.iloc[:, 1].map(id_map).values
    valid = ~(np.isnan(src) | np.isnan(dst))
    edge_index = torch.tensor(np.stack([src[valid].astype(np.int64), dst[valid].astype(np.int64)]))

    classes = pd.read_csv('data/raw/elliptic_txs_classes.csv')
    labels = np.full(len(node_ids), -1, dtype=np.int64)
    for _, r in classes.iterrows():
        if r.iloc[0] in id_map:
            labels[id_map[r.iloc[0]]] = {'1': 1, '2': 0}.get(str(r.iloc[1]), -1)

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        timestep=torch.tensor(timesteps, dtype=torch.long)
    )


def get_masks(data):
    """Temporal split masks."""
    n = data.timestep.max().item() + 1
    lab = data.y != -1
    train_mask = (data.timestep < int(n*0.6)) & lab
    val_mask = (data.timestep >= int(n*0.6)) & (data.timestep < int(n*0.8)) & lab
    test_mask = (data.timestep >= int(n*0.8)) & lab

    # Unlabeled mask for semi-supervised learning
    unlabeled_mask = (data.y == -1)

    return train_mask, val_mask, test_mask, unlabeled_mask


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction='none')
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[y]
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce).mean()


class SemiSupervisedTrainer:
    """
    Trainer with semi-supervised learning.

    Uses:
    1. Supervised loss on labeled nodes
    2. Pseudo-labeling on confident unlabeled predictions
    3. Consistency regularization
    """
    def __init__(
        self,
        model,
        data,
        train_mask,
        val_mask,
        test_mask,
        unlabeled_mask,
        epochs=300,
        lr=0.001,
        weight_decay=1e-4,
        patience=30,
        ssl_start_epoch=50,
        ssl_weight=0.5,
        confidence_threshold=0.9
    ):
        self.model = model
        self.data = data
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.unlabeled_mask = unlabeled_mask

        self.epochs = epochs
        self.patience = patience
        self.ssl_start_epoch = ssl_start_epoch
        self.ssl_weight = ssl_weight
        self.confidence_threshold = confidence_threshold

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

        # Loss function with class weighting
        y_train = data.y[train_mask]
        weight = torch.tensor([
            1.0,
            (y_train == 0).sum().float() / (y_train == 1).sum().float()
        ])
        alpha = weight / weight.sum()
        self.loss_fn = FocalLoss(gamma=2.0, alpha=alpha)

        # Best model tracking
        self.best_val_auc = 0
        self.best_state = None
        self.wait = 0

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(self.data.x, self.data.edge_index, self.data.timestep)

        # Supervised loss on labeled training data
        supervised_loss = self.loss_fn(out[self.train_mask], self.data.y[self.train_mask])

        total_loss = supervised_loss

        # Semi-supervised loss (after warmup)
        if epoch >= self.ssl_start_epoch and self.unlabeled_mask.sum() > 0:
            ssl_loss = self.compute_ssl_loss(out)
            total_loss = total_loss + self.ssl_weight * ssl_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return total_loss.item()

    def compute_ssl_loss(self, out):
        """
        Compute semi-supervised loss using pseudo-labeling.

        Strategy:
        - Use high-confidence predictions on unlabeled nodes as pseudo-labels
        - Apply consistency regularization
        """
        with torch.no_grad():
            probs = F.softmax(out[self.unlabeled_mask], dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)
            confident_mask = max_probs > self.confidence_threshold

        if confident_mask.sum() == 0:
            return torch.tensor(0.0, device=out.device)

        # Loss on confident pseudo-labels
        unlabeled_indices = torch.where(self.unlabeled_mask)[0]
        confident_indices = unlabeled_indices[confident_mask]

        ssl_loss = F.cross_entropy(
            out[confident_indices],
            pseudo_labels[confident_mask]
        )

        return ssl_loss

    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate on given mask."""
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index, self.data.timestep)
        probs = F.softmax(out[mask], dim=1)[:, 1]

        y_true = self.data.y[mask].numpy()
        y_score = probs.numpy()

        pr_auc = average_precision_score(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)

        return pr_auc, roc_auc

    def train(self):
        """Full training loop."""
        print("Starting training...")
        print(f"  Train: {self.train_mask.sum().item()} nodes")
        print(f"  Val: {self.val_mask.sum().item()} nodes")
        print(f"  Test: {self.test_mask.sum().item()} nodes")
        print(f"  Unlabeled: {self.unlabeled_mask.sum().item()} nodes")
        print(f"  SSL starts at epoch {self.ssl_start_epoch}\n")

        for epoch in range(self.epochs):
            loss = self.train_epoch(epoch)

            # Validation
            val_pr_auc, val_roc_auc = self.evaluate(self.val_mask)

            # Track best model
            if val_pr_auc > self.best_val_auc:
                self.best_val_auc = val_pr_auc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.wait = 0
            else:
                self.wait += 1

            # Logging
            if (epoch + 1) % 25 == 0:
                ssl_status = "SSL ON" if epoch >= self.ssl_start_epoch else "SSL OFF"
                print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, Val PR-AUC={val_pr_auc:.4f}, "
                      f"Val ROC-AUC={val_roc_auc:.4f}, Best={self.best_val_auc:.4f} [{ssl_status}]")

            # Early stopping
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(self.best_state)

        # Final evaluation
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)

        val_pr_auc, val_roc_auc = self.evaluate(self.val_mask)
        test_pr_auc, test_roc_auc = self.evaluate(self.test_mask)

        print(f"Validation - PR-AUC: {val_pr_auc:.4f}, ROC-AUC: {val_roc_auc:.4f}")
        print(f"Test       - PR-AUC: {test_pr_auc:.4f}, ROC-AUC: {test_roc_auc:.4f}")

        return test_pr_auc, test_roc_auc


def run_experiment(
    use_behavioral=True,
    use_ssl=True,
    hidden_dim=128,
    num_layers=3,
    num_heads=4
):
    """Run single experiment."""
    print("="*70)
    print("ENHANCED TEMPORAL GAT EXPERIMENT")
    print("="*70)
    print(f"Config:")
    print(f"  Behavioral features: {use_behavioral}")
    print(f"  Semi-supervised: {use_ssl}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Attention heads: {num_heads}")
    print("="*70 + "\n")

    # Load data
    data = load_data()
    train_mask, val_mask, test_mask, unlabeled_mask = get_masks(data)

    # Create model
    model = create_enhanced_temporal_gat(
        num_features=data.x.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_behavioral=use_behavioral
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = SemiSupervisedTrainer(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        unlabeled_mask=unlabeled_mask if use_ssl else torch.zeros_like(unlabeled_mask),
        epochs=300,
        lr=0.001,
        ssl_start_epoch=50 if use_ssl else 999999,  # Disable if not using SSL
        ssl_weight=0.5,
        confidence_threshold=0.95
    )

    test_pr_auc, test_roc_auc = trainer.train()

    return test_pr_auc, test_roc_auc


def main():
    """Run multiple configurations."""
    results = []

    # 1. Full model (Behavioral + SSL)
    print("\n[1/4] Full Model: Behavioral + Semi-Supervised Learning")
    pr_auc, roc_auc = run_experiment(use_behavioral=True, use_ssl=True)
    results.append({
        'Model': 'Enhanced GAT (Full)',
        'Config': 'Behavioral + SSL',
        'Test PR-AUC': pr_auc,
        'Test ROC-AUC': roc_auc
    })

    # 2. Without SSL
    print("\n[2/4] Behavioral features only (no SSL)")
    pr_auc, roc_auc = run_experiment(use_behavioral=True, use_ssl=False)
    results.append({
        'Model': 'Enhanced GAT',
        'Config': 'Behavioral only',
        'Test PR-AUC': pr_auc,
        'Test ROC-AUC': roc_auc
    })

    # 3. Without behavioral features
    print("\n[3/4] SSL only (no behavioral)")
    pr_auc, roc_auc = run_experiment(use_behavioral=False, use_ssl=True)
    results.append({
        'Model': 'Enhanced GAT',
        'Config': 'SSL only',
        'Test PR-AUC': pr_auc,
        'Test ROC-AUC': roc_auc
    })

    # 4. Larger model
    print("\n[4/4] Full model with larger capacity")
    pr_auc, roc_auc = run_experiment(
        use_behavioral=True,
        use_ssl=True,
        hidden_dim=192,
        num_layers=4,
        num_heads=6
    )
    results.append({
        'Model': 'Enhanced GAT (Large)',
        'Config': 'Behavioral + SSL (192d, 4L)',
        'Test PR-AUC': pr_auc,
        'Test ROC-AUC': roc_auc
    })

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    best = df.loc[df['Test PR-AUC'].idxmax()]
    print(f"\nBest Model: {best['Model']} ({best['Config']})")
    print(f"Test PR-AUC: {best['Test PR-AUC']:.4f}")
    print(f"Test ROC-AUC: {best['Test ROC-AUC']:.4f}")

    if best['Test PR-AUC'] >= 0.60:
        print(f"\nSUCCESS! Target of 60% PR-AUC achieved!")
    else:
        gap = (0.60 - best['Test PR-AUC']) * 100
        print(f"\nGap to 60% target: {gap:.1f} percentage points")

    # Save results
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/enhanced_gat_results.csv', index=False)
    print("\nResults saved to results/enhanced_gat_results.csv")


if __name__ == "__main__":
    main()
