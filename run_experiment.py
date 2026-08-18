"""
Main Experiment Script
======================

Run the full AML detection experiment pipeline:
1. Load and preprocess Elliptic dataset
2. Train baseline models (Logistic Regression, GraphSAGE)
3. Train advanced models (Temporal GNN, Edge-Aware GNN)
4. Evaluate and compare all models
5. Generate explanations for high-risk predictions

Usage:
    python run_experiment.py --config config.yaml
    python run_experiment.py --model graphsage
    python run_experiment.py --model all
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import set_seed, get_device, load_config, save_results, Timer
from src.data.elliptic_dataset import load_elliptic_data
from src.data.preprocessing import (
    create_temporal_splits,
    compute_class_weights,
    build_edge_features,
)
from src.models.baseline import LogisticRegressionBaseline
from src.models.graphsage import create_graphsage_model
from src.models.temporal_gnn import create_temporal_gnn
from src.models.edge_aware_gnn import create_edge_aware_gnn
from src.training.trainer import train_model, evaluate_model
from src.training.metrics import compute_metrics, plot_comparison, plot_precision_recall_curve
from src.explainability.gnn_explainer import explain_batch, get_high_risk_nodes


def run_logistic_regression(data, train_mask, val_mask, test_mask, seed=42):
    """Train and evaluate Logistic Regression baseline."""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION BASELINE")
    print("="*60)

    # Convert to numpy
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    train_idx = train_mask.cpu().numpy()
    val_idx = val_mask.cpu().numpy()
    test_idx = test_mask.cpu().numpy()

    # Train
    model = LogisticRegressionBaseline(random_state=seed)
    model.fit(X[train_idx], y[train_idx])

    # Evaluate
    print("\n--- Validation Results ---")
    val_results = model.evaluate(X[val_idx], y[val_idx])

    print("\n--- Test Results ---")
    test_results = model.evaluate(X[test_idx], y[test_idx])

    return model, {
        'val_pr_auc': val_results['pr_auc'],
        'test_pr_auc': test_results['pr_auc'],
        'test_metrics': test_results,
    }


def run_graphsage(data, train_mask, val_mask, test_mask, config, device, seed=42):
    """Train and evaluate Static GraphSAGE."""
    print("\n" + "="*60)
    print("STATIC GRAPHSAGE")
    print("="*60)

    # Get model config
    model_config = config.get('model', {}).get('graphsage', {})

    # Create model
    model = create_graphsage_model(
        num_features=data.x.shape[1],
        hidden_dim=model_config.get('hidden_dim', 128),
        num_classes=2,
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.5),
        device=device,
    )

    # Compute class weights
    class_weights = compute_class_weights(data.y, train_mask)

    # Train
    training_config = config.get('training', {})
    model, results = train_model(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        class_weights=class_weights,
        num_epochs=training_config.get('num_epochs', 200),
        patience=training_config.get('early_stopping_patience', 20),
        learning_rate=training_config.get('learning_rate', 0.01),
        device=device,
        seed=seed,
    )

    return model, results


def run_temporal_gnn(data, train_mask, val_mask, test_mask, config, device, seed=42):
    """Train and evaluate Temporal GNN."""
    print("\n" + "="*60)
    print("TEMPORAL GNN")
    print("="*60)

    # Get model config
    model_config = config.get('model', {}).get('temporal_gnn', {})

    # Create model
    model = create_temporal_gnn(
        num_features=data.x.shape[1],
        hidden_dim=model_config.get('hidden_dim', 128),
        num_classes=2,
        time_dim=model_config.get('time_dim', 32),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.5),
        use_memory=True,
        device=device,
    )

    # Compute class weights
    class_weights = compute_class_weights(data.y, train_mask)

    # Train
    training_config = config.get('training', {})
    model, results = train_model(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        class_weights=class_weights,
        num_epochs=training_config.get('num_epochs', 200),
        patience=training_config.get('early_stopping_patience', 20),
        learning_rate=training_config.get('learning_rate', 0.01),
        device=device,
        seed=seed,
    )

    return model, results


def run_edge_aware_gnn(data, train_mask, val_mask, test_mask, edge_attr, config, device, seed=42):
    """Train and evaluate Edge-Aware GNN."""
    print("\n" + "="*60)
    print("EDGE-AWARE GNN")
    print("="*60)

    # Get model config
    model_config = config.get('model', {}).get('temporal_gnn', {})

    # Create model
    model = create_edge_aware_gnn(
        num_features=data.x.shape[1],
        edge_features=edge_attr.shape[1],
        hidden_dim=model_config.get('hidden_dim', 128),
        num_classes=2,
        num_layers=model_config.get('num_layers', 2),
        heads=model_config.get('num_heads', 4),
        dropout=model_config.get('dropout', 0.5),
        device=device,
    )

    # Compute class weights
    class_weights = compute_class_weights(data.y, train_mask)

    # Train
    training_config = config.get('training', {})
    model, results = train_model(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        class_weights=class_weights,
        edge_attr=edge_attr,
        num_epochs=training_config.get('num_epochs', 200),
        patience=training_config.get('early_stopping_patience', 20),
        learning_rate=training_config.get('learning_rate', 0.01),
        device=device,
        seed=seed,
    )

    return model, results


def main():
    parser = argparse.ArgumentParser(description="Run AML Detection Experiment")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default='all',
                       choices=['logistic', 'graphsage', 'temporal', 'edge_aware', 'all'],
                       help='Model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()

    # Setup
    print("="*60)
    print("EXPLAINABLE TEMPORAL GNN FOR ANTI-MONEY LAUNDERING")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set seed
    set_seed(args.seed)

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file {args.config} not found, using defaults")
        config = {}

    # Get device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    try:
        with Timer("Data loading"):
            data, metadata = load_elliptic_data(args.data_dir, device=str(device))
            print(f"\nDataset loaded successfully!")
            print(f"  Nodes: {metadata['num_nodes']:,}")
            print(f"  Edges: {metadata['num_edges']:,}")
            print(f"  Features: {metadata['num_features']}")
            print(f"  Timesteps: {metadata['num_timesteps']}")
            print(f"  Labeled: {metadata['num_labeled']:,}")
            print(f"  Illicit: {metadata['num_illicit']:,}")
            print(f"  Licit: {metadata['num_licit']:,}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the Elliptic dataset from:")
        print("https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        print(f"\nExtract the files to: {os.path.join(args.data_dir, 'raw')}")
        return

    # Create splits
    print("\n" + "="*60)
    print("CREATING DATA SPLITS")
    print("="*60)

    training_config = config.get('training', {})
    train_mask, val_mask, test_mask = create_temporal_splits(
        data,
        train_ratio=training_config.get('train_ratio', 0.6),
        val_ratio=training_config.get('val_ratio', 0.2),
        test_ratio=training_config.get('test_ratio', 0.2),
        seed=args.seed,
    )

    # Build edge features
    print("\n" + "="*60)
    print("BUILDING EDGE FEATURES")
    print("="*60)

    edge_config = config.get('model', {}).get('edge_features', {})
    edge_attr = build_edge_features(
        data,
        use_direction=edge_config.get('use_direction', True),
        use_time_gap=edge_config.get('use_time_gap', True),
        use_degree_dynamics=edge_config.get('use_degree_dynamics', True),
        use_activity_rate=edge_config.get('use_activity_rate', True),
    )

    # Store results
    all_results = {}

    # Run experiments
    if args.model in ['logistic', 'all']:
        _, lr_results = run_logistic_regression(data, train_mask, val_mask, test_mask, args.seed)
        all_results['Logistic Regression'] = lr_results

    if args.model in ['graphsage', 'all']:
        gs_model, gs_results = run_graphsage(
            data, train_mask, val_mask, test_mask, config, device, args.seed
        )
        all_results['GraphSAGE'] = gs_results

    if args.model in ['temporal', 'all']:
        tgnn_model, tgnn_results = run_temporal_gnn(
            data, train_mask, val_mask, test_mask, config, device, args.seed
        )
        all_results['Temporal GNN'] = tgnn_results

    if args.model in ['edge_aware', 'all']:
        eagnn_model, eagnn_results = run_edge_aware_gnn(
            data, train_mask, val_mask, test_mask, edge_attr, config, device, args.seed
        )
        all_results['Edge-Aware GNN'] = eagnn_results

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    summary_data = []
    for model_name, results in all_results.items():
        if 'test_metrics' in results:
            metrics = results['test_metrics']
            pr_auc = metrics.get('pr_auc', results.get('test_pr_auc', 0))
        else:
            pr_auc = results.get('test_pr_auc', 0)

        summary_data.append({
            'Model': model_name,
            'PR-AUC': pr_auc,
        })
        print(f"{model_name}: PR-AUC = {pr_auc:.4f}")

    # Save results
    save_results(all_results, args.output_dir, f"experiment_{args.seed}")

    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(args.output_dir, f"summary_{args.seed}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
