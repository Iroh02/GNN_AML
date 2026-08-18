
"""
Evaluation Metrics for AML Detection
=====================================

Metrics focused on imbalanced classification:
- PR-AUC (Area Under Precision-Recall Curve)
- Precision@K
- Precision-Recall curves
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Tuple, Optional
import warnings


def compute_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class

    Returns:
        PR-AUC score
    """
    # Filter out unknown labels
    mask = y_true >= 0
    y_true = y_true[mask]
    y_proba = y_proba[mask]

    if len(np.unique(y_true)) < 2:
        warnings.warn("Only one class present in y_true. PR-AUC is undefined.")
        return 0.0

    return average_precision_score(y_true, y_proba)


def compute_precision_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k_values: List[int] = [100, 500, 1000]
) -> Dict[int, float]:
    """
    Compute Precision@K for multiple K values.

    This metric is operationally relevant in AML settings where
    analysts investigate only a limited number of alerts.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        k_values: List of K values to evaluate

    Returns:
        Dictionary mapping K to Precision@K
    """
    # Filter out unknown labels
    mask = y_true >= 0
    y_true = y_true[mask]
    y_proba = y_proba[mask]

    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_proba)[::-1]
    sorted_labels = y_true[sorted_indices]

    results = {}
    for k in k_values:
        if k > len(sorted_labels):
            k = len(sorted_labels)
        precision_k = sorted_labels[:k].sum() / k
        results[k] = precision_k

    return results


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        threshold: Classification threshold

    Returns:
        Dictionary with all metrics
    """
    # Filter out unknown labels
    mask = y_true >= 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_proba is not None:
        y_proba = y_proba[mask]

    metrics = {}

    # Basic metrics
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = tp
        metrics['false_positives'] = fp
        metrics['true_negatives'] = tn
        metrics['false_negatives'] = fn
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Probability-based metrics
    if y_proba is not None:
        if len(np.unique(y_true)) >= 2:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

            # Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            metrics['pr_curve'] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }

            # Precision@K
            metrics['precision_at_k'] = compute_precision_at_k(y_true, y_proba)
        else:
            metrics['pr_auc'] = 0.0
            metrics['roc_auc'] = 0.0
    else:
        metrics['pr_auc'] = 0.0

    return metrics


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: str = "Model",
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        label: Label for the curve
        ax: Matplotlib axes (optional)
        color: Line color (optional)

    Returns:
        Figure and axes
    """
    # Filter out unknown labels
    mask = y_true >= 0
    y_true = y_true[mask]
    y_proba = y_proba[mask]

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.plot(recall, precision, label=f"{label} (PR-AUC = {pr_auc:.3f})", color=color)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', label='Random')

    return fig, ax


def plot_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models.

    Args:
        results: Dictionary mapping model names to their evaluation results
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # PR curves
    ax1 = axes[0]
    for (name, res), color in zip(results.items(), colors):
        if 'pr_curve' in res:
            pr = res['pr_curve']
            pr_auc = res.get('pr_auc', 0)
            ax1.plot(
                pr['recall'],
                pr['precision'],
                label=f"{name} (PR-AUC = {pr_auc:.3f})",
                color=color
            )

    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Bar plot of metrics
    ax2 = axes[1]
    metric_names = ['pr_auc', 'precision', 'recall', 'f1']
    x = np.arange(len(metric_names))
    width = 0.8 / len(results)

    for i, (name, res) in enumerate(results.items()):
        values = [res.get(m, 0) for m in metric_names]
        ax2.bar(x + i * width, values, width, label=name, color=colors[i])

    ax2.set_ylabel('Score')
    ax2.set_title('Model Comparison')
    ax2.set_xticks(x + width * (len(results) - 1) / 2)
    ax2.set_xticklabels(['PR-AUC', 'Precision', 'Recall', 'F1'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = ['Licit', 'Illicit']
):
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for each class
    """
    # Filter out unknown labels
    mask = y_true >= 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=target_names))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)


def statistical_significance_test(
    results_1: List[float],
    results_2: List[float],
    test: str = 'paired_t'
) -> Tuple[float, float]:
    """
    Perform statistical significance test between two models.

    Args:
        results_1: List of scores from model 1 (multiple runs)
        results_2: List of scores from model 2 (multiple runs)
        test: Type of test ('paired_t' or 'wilcoxon')

    Returns:
        Tuple of (test statistic, p-value)
    """
    from scipy import stats

    if test == 'paired_t':
        stat, pval = stats.ttest_rel(results_1, results_2)
    elif test == 'wilcoxon':
        stat, pval = stats.wilcoxon(results_1, results_2)
    else:
        raise ValueError(f"Unknown test: {test}")

    return stat, pval


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.randint(0, 2, n_samples)
    y_proba = np.random.rand(n_samples)
    y_pred = (y_proba > 0.5).astype(int)

    # Test metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print(f"\nMetrics:")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    # Test Precision@K
    p_at_k = compute_precision_at_k(y_true, y_proba)
    print(f"\nPrecision@K:")
    for k, p in p_at_k.items():
        print(f"  P@{k}: {p:.4f}")

    print("\nMetrics test passed!")
