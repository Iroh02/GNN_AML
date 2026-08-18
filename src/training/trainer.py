"""
Training Pipeline for GNN Models
=================================

Unified training and evaluation pipeline for all models
including Logistic Regression, GraphSAGE, Temporal GNN,
and Edge-Aware GNN.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
from tqdm import tqdm
import time

from .metrics import compute_metrics, compute_pr_auc


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class Trainer:
    """
    Unified trainer for GNN models.

    Handles training, validation, and evaluation with:
    - Class-weighted loss for imbalanced data
    - Early stopping
    - Learning rate scheduling
    - Metrics tracking

    Args:
        model: PyTorch model to train
        device: Device for training
        learning_rate: Learning rate
        weight_decay: L2 regularization
        class_weights: Class weights for loss function
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 0.01,
        weight_decay: float = 0.0005,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function with class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            # verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_pr_auc': [],
            'val_pr_auc': [],
        }

    def train_epoch(
        self,
        data: Data,
        train_mask: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            data: PyG Data object
            train_mask: Mask for training nodes
            edge_attr: Optional edge attributes

        Returns:
            Tuple of (loss, pr_auc)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        if edge_attr is not None and hasattr(self.model, 'forward'):
            # Edge-aware model
            if hasattr(data, 'timestep'):
                out = self.model(data.x, data.edge_index, edge_attr, data.timestep)
            else:
                out = self.model(data.x, data.edge_index, edge_attr)
        elif hasattr(data, 'timestep') and hasattr(self.model, 'forward'):
            # Temporal model
            out = self.model(data.x, data.edge_index, data.timestep)
        else:
            # Standard model
            out = self.model(data.x, data.edge_index)

        # Compute loss only on training nodes
        loss = self.criterion(out[train_mask], data.y[train_mask])

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            probs = F.softmax(out[train_mask], dim=1)[:, 1].cpu().numpy()
            labels = data.y[train_mask].cpu().numpy()
            pr_auc = compute_pr_auc(labels, probs)

        return loss.item(), pr_auc

    @torch.no_grad()
    def evaluate(
        self,
        data: Data,
        mask: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Evaluate model on masked nodes.

        Args:
            data: PyG Data object
            mask: Mask for evaluation nodes
            edge_attr: Optional edge attributes

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        # Forward pass
        if edge_attr is not None:
            if hasattr(data, 'timestep'):
                out = self.model(data.x, data.edge_index, edge_attr, data.timestep)
            else:
                out = self.model(data.x, data.edge_index, edge_attr)
        elif hasattr(data, 'timestep') and hasattr(self.model, 'forward'):
            try:
                out = self.model(data.x, data.edge_index, data.timestep)
            except TypeError:
                out = self.model(data.x, data.edge_index)
        else:
            out = self.model(data.x, data.edge_index)

        # Compute loss
        loss = self.criterion(out[mask], data.y[mask]).item()

        # Get predictions and probabilities
        probs = F.softmax(out[mask], dim=1)
        pred = probs.argmax(dim=1)

        # Compute metrics
        metrics = compute_metrics(
            y_true=data.y[mask].cpu().numpy(),
            y_pred=pred.cpu().numpy(),
            y_proba=probs[:, 1].cpu().numpy()
        )
        metrics['loss'] = loss

        return metrics

    def train(
        self,
        data: Data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        num_epochs: int = 200,
        patience: int = 20,
        edge_attr: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Full training loop.

        Args:
            data: PyG Data object
            train_mask: Training node mask
            val_mask: Validation node mask
            num_epochs: Maximum epochs
            patience: Early stopping patience
            edge_attr: Optional edge attributes
            verbose: Print progress

        Returns:
            Training history
        """
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)

        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)

        early_stopping = EarlyStopping(patience=patience, mode='max')
        best_val_pr_auc = 0
        best_model_state = None

        pbar = tqdm(range(num_epochs), disable=not verbose, desc="Training")

        for epoch in pbar:
            # Training
            train_loss, train_pr_auc = self.train_epoch(data, train_mask, edge_attr)

            # Validation
            val_metrics = self.evaluate(data, val_mask, edge_attr)
            val_loss = val_metrics['loss']
            val_pr_auc = val_metrics['pr_auc']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_pr_auc'].append(train_pr_auc)
            self.history['val_pr_auc'].append(val_pr_auc)

            # Update learning rate
            self.scheduler.step(val_pr_auc)

            # Save best model
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_pr_auc': f'{val_pr_auc:.4f}',
                'best': f'{best_val_pr_auc:.4f}'
            })

            # Early stopping
            if early_stopping(val_pr_auc):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


def train_model(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    edge_attr: Optional[torch.Tensor] = None,
    num_epochs: int = 200,
    patience: int = 20,
    learning_rate: float = 0.01,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[nn.Module, Dict]:
    """
    Train a model and return results.

    Args:
        model: PyTorch model
        data: PyG Data object
        train_mask: Training mask
        val_mask: Validation mask
        test_mask: Test mask
        class_weights: Optional class weights
        edge_attr: Optional edge attributes
        num_epochs: Maximum epochs
        patience: Early stopping patience
        learning_rate: Learning rate
        device: Training device
        seed: Random seed

    Returns:
        Tuple of (trained model, results dictionary)
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        class_weights=class_weights,
    )

    # Train
    print("Starting training...")
    start_time = time.time()

    history = trainer.train(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        num_epochs=num_epochs,
        patience=patience,
        edge_attr=edge_attr,
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(data, test_mask, edge_attr)

    results = {
        'history': history,
        'test_metrics': test_metrics,
        'training_time': training_time,
    }

    print(f"\nTest Results:")
    print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"  Precision: {test_metrics.get('precision', 0):.4f}")
    print(f"  Recall: {test_metrics.get('recall', 0):.4f}")
    print(f"  F1: {test_metrics.get('f1', 0):.4f}")

    return model, results


def evaluate_model(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate a trained model.

    Args:
        model: Trained model
        data: PyG Data object
        mask: Evaluation mask
        edge_attr: Optional edge attributes
        device: Device for evaluation

    Returns:
        Evaluation metrics
    """
    model = model.to(device)
    model.eval()

    data = data.to(device)
    mask = mask.to(device)

    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    with torch.no_grad():
        if edge_attr is not None:
            if hasattr(data, 'timestep'):
                out = model(data.x, data.edge_index, edge_attr, data.timestep)
            else:
                out = model(data.x, data.edge_index, edge_attr)
        elif hasattr(data, 'timestep'):
            try:
                out = model(data.x, data.edge_index, data.timestep)
            except TypeError:
                out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index)

        probs = F.softmax(out[mask], dim=1)
        pred = probs.argmax(dim=1)

        metrics = compute_metrics(
            y_true=data.y[mask].cpu().numpy(),
            y_pred=pred.cpu().numpy(),
            y_proba=probs[:, 1].cpu().numpy()
        )

    return metrics


if __name__ == "__main__":
    # Test training pipeline
    print("Testing training pipeline...")

    from ..models.graphsage import create_graphsage_model

    # Create dummy data
    num_nodes = 1000
    num_features = 166
    num_edges = 5000

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))
    timestep = torch.randint(0, 49, (num_nodes,))

    data = Data(x=x, edge_index=edge_index, y=y, timestep=timestep)

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:600] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[600:800] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[800:] = True

    # Create model
    model = create_graphsage_model(num_features=num_features)

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, results = train_model(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_epochs=10,
        patience=5,
        device=device,
    )

    print("\nTraining pipeline test passed!")
