"""Training and evaluation modules."""

from .trainer import Trainer, train_model, evaluate_model
from .metrics import (
    compute_pr_auc,
    compute_precision_at_k,
    compute_metrics,
    plot_precision_recall_curve,
)

__all__ = [
    "Trainer",
    "train_model",
    "evaluate_model",
    "compute_pr_auc",
    "compute_precision_at_k",
    "compute_metrics",
    "plot_precision_recall_curve",
]
