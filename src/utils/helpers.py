"""
Utility Functions
=================

Helper functions for reproducibility, configuration,
and result management.
"""

import os
import random
import numpy as np
import torch
import yaml
import json
import pickle
from typing import Dict, Any, Optional
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_gpu: Whether to prefer GPU if available

    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(
    results: Dict[str, Any],
    save_dir: str = "results",
    name: Optional[str] = None,
    format: str = "json"
):
    """
    Save experiment results.

    Args:
        results: Results dictionary
        save_dir: Directory to save results
        name: Optional name for the results file
        format: Save format ('json' or 'pickle')
    """
    os.makedirs(save_dir, exist_ok=True)

    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"results_{timestamp}"

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    if format == "json":
        path = os.path.join(save_dir, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(convert_for_json(results), f, indent=2)
    elif format == "pickle":
        path = os.path.join(save_dir, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Results saved to {path}")
    return path


def load_results(path: str) -> Dict[str, Any]:
    """
    Load experiment results.

    Args:
        path: Path to results file

    Returns:
        Results dictionary
    """
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown file format: {path}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics for printing.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {name}: {value:.4f}")
        elif isinstance(value, (dict, list, np.ndarray)):
            continue  # Skip complex objects
        else:
            lines.append(f"  {name}: {value}")
    return "\n".join(lines)


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        self.end = datetime.now()
        duration = (self.end - self.start).total_seconds()
        if self.name:
            print(f"{self.name}: {duration:.2f} seconds")

    @property
    def duration(self):
        if self.start and self.end:
            return (self.end - self.start).total_seconds()
        return None


def create_experiment_dir(base_dir: str = "experiments") -> str:
    """
    Create a timestamped experiment directory.

    Args:
        base_dir: Base directory for experiments

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    for subdir in ["checkpoints", "results", "plots"]:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    print(f"Created experiment directory: {exp_dir}")
    return exp_dir


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test seed setting
    set_seed(42)

    # Test device detection
    device = get_device()

    # Test timer
    with Timer("Test operation"):
        import time
        time.sleep(0.1)

    print("\nUtility tests passed!")
