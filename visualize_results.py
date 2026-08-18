"""
Visualize Model Comparison Results
===================================

Creates publication-quality visualizations of model performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def load_results():
    """Load results from CSV files."""
    results = {}

    if os.path.exists('results/complete_comparison.csv'):
        results['complete'] = pd.read_csv('results/complete_comparison.csv')
    elif os.path.exists('results/final_comparison.csv'):
        results['final'] = pd.read_csv('results/final_comparison.csv')

    if os.path.exists('results/enhanced_gat_results.csv'):
        results['enhanced'] = pd.read_csv('results/enhanced_gat_results.csv')

    return results


def plot_model_comparison(results_dict):
    """Create comprehensive comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Use whichever results are available
    if 'complete' in results_dict:
        df = results_dict['complete']
    elif 'final' in results_dict:
        df = results_dict['final']
    else:
        print("No comparison results found!")
        return

    # Plot 1: PR-AUC comparison
    ax1 = axes[0, 0]
    models = df['Model'].values
    pr_aucs = df['PR-AUC'].values if 'PR-AUC' in df.columns else df['Test PR-AUC'].values

    colors = ['#d62728' if pr < 0.4 else '#ff7f0e' if pr < 0.5 else '#2ca02c' for pr in pr_aucs]
    bars = ax1.barh(models, pr_aucs, color=colors, alpha=0.8, edgecolor='black')
    ax1.axvline(x=0.60, color='red', linestyle='--', linewidth=2, label='Target (60%)')
    ax1.set_xlabel('PR-AUC', fontweight='bold')
    ax1.set_title('Model Performance Comparison (PR-AUC)', fontweight='bold', pad=15)
    ax1.legend()
    ax1.set_xlim([0, max(pr_aucs) * 1.1])

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, pr_aucs)):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')

    # Plot 2: Improvement over baseline
    ax2 = axes[0, 1]
    if 'Improvement vs LR (%)' in df.columns:
        improvements = df['Improvement vs LR (%)'].values
    elif 'Improvement vs LR' in df.columns:
        improvements = df['Improvement vs LR'].values
    else:
        baseline = pr_aucs[0]
        improvements = ((pr_aucs - baseline) / baseline * 100)

    colors2 = ['#2ca02c' if imp > 150 else '#ff7f0e' if imp > 0 else '#d62728' for imp in improvements]
    bars2 = ax2.barh(models, improvements, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Improvement vs Baseline (%)', fontweight='bold')
    ax2.set_title('Relative Improvement', fontweight='bold', pad=15)

    for i, (bar, val) in enumerate(zip(bars2, improvements)):
        ax2.text(val + 5, i, f'+{val:.1f}%', va='center', fontweight='bold')

    # Plot 3: Model progression
    ax3 = axes[1, 0]
    x_pos = range(len(models))
    ax3.plot(x_pos, pr_aucs, marker='o', markersize=10, linewidth=2, color='#1f77b4', label='PR-AUC')
    ax3.axhline(y=0.60, color='red', linestyle='--', linewidth=2, label='Target (60%)')
    ax3.fill_between(x_pos, 0, pr_aucs, alpha=0.3, color='#1f77b4')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylabel('PR-AUC', fontweight='bold')
    ax3.set_xlabel('Model Evolution', fontweight='bold')
    ax3.set_title('Model Progression Over Time', fontweight='bold', pad=15)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    best_idx = pr_aucs.argmax()
    best_model = models[best_idx]
    best_pr = pr_aucs[best_idx]

    summary_text = f"""
MODEL PERFORMANCE SUMMARY
{'='*50}

BEST MODEL: {best_model}
Test PR-AUC: {best_pr:.4f}

BASELINE COMPARISON:
  Logistic Regression: {pr_aucs[0]:.4f}
  Best Model: {best_pr:.4f}
  Improvement: +{improvements[best_idx]:.1f}%

TARGET ACHIEVEMENT:
  Target: 60.0% PR-AUC
  Best: {best_pr*100:.1f}% PR-AUC
  {'✓ TARGET REACHED!' if best_pr >= 0.60 else f'Gap: {(0.60-best_pr)*100:.1f} percentage points'}

KEY INNOVATIONS:
  1. Graph Neural Networks (+{(pr_aucs[1]-pr_aucs[0])/pr_aucs[0]*100:.0f}% over baseline)
  2. Temporal Encoding (+{(pr_aucs[2]-pr_aucs[1])/pr_aucs[1]*100:.0f}% over static)
  3. Relative Time (fixes leakage)
  4. Attention + Behavioral Features
  5. Semi-Supervised Learning

CONCLUSION:
  {'SUCCESS! GNN approach significantly improves AML detection.' if best_pr >= 0.60 else 'Strong improvement over baseline. Further tuning needed for 60% target.'}
"""

    ax4.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results/model_comparison_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: results/model_comparison_visualization.png")


def plot_enhanced_gat_ablation(results_dict):
    """Plot ablation study for Enhanced GAT."""
    if 'enhanced' not in results_dict:
        print("No Enhanced GAT ablation results found.")
        return

    df = results_dict['enhanced']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PR-AUC comparison
    configs = df['Config'].values
    pr_aucs = df['Test PR-AUC'].values

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(configs)))
    bars = ax1.bar(range(len(configs)), pr_aucs, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.set_ylabel('Test PR-AUC', fontweight='bold')
    ax1.set_title('Enhanced GAT: Ablation Study', fontweight='bold', pad=15)
    ax1.axhline(y=0.60, color='red', linestyle='--', linewidth=2, label='Target')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, pr_aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # ROC-AUC comparison
    if 'Test ROC-AUC' in df.columns:
        roc_aucs = df['Test ROC-AUC'].values
        bars2 = ax2.bar(range(len(configs)), roc_aucs, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.set_ylabel('Test ROC-AUC', fontweight='bold')
        ax2.set_title('ROC-AUC Performance', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars2, roc_aucs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/enhanced_gat_ablation.png', dpi=300, bbox_inches='tight')
    print("Saved: results/enhanced_gat_ablation.png")


def create_summary_table():
    """Create a formatted summary table."""
    results = load_results()

    if 'complete' in results:
        df = results['complete']
    elif 'final' in results:
        df = results['final']
    else:
        print("No results to summarize!")
        return

    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON - SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    best_idx = df['PR-AUC'].idxmax() if 'PR-AUC' in df.columns else df['Test PR-AUC'].idxmax()
    best = df.iloc[best_idx]
    print(f"\nBEST MODEL: {best['Model']}")
    print(f"Performance: {best['PR-AUC'] if 'PR-AUC' in df.columns else best['Test PR-AUC']:.4f} PR-AUC")


def main():
    """Generate all visualizations."""
    print("Loading results...")
    results = load_results()

    if not results:
        print("No results found! Please run experiments first.")
        return

    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)

    # Main comparison plot
    print("\n1. Creating model comparison visualization...")
    plot_model_comparison(results)

    # Enhanced GAT ablation
    if 'enhanced' in results:
        print("\n2. Creating Enhanced GAT ablation study...")
        plot_enhanced_gat_ablation(results)

    # Summary table
    print("\n3. Generating summary table...")
    create_summary_table()

    print("\n" + "="*60)
    print("  VISUALIZATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/model_comparison_visualization.png")
    if 'enhanced' in results:
        print("  - results/enhanced_gat_ablation.png")


if __name__ == "__main__":
    import numpy as np  # For colormap
    main()
