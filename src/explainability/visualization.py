"""
Visualization Module for GNN Explanations
==========================================

Visualize explanations including:
- Subgraph visualizations centered on suspicious nodes
- Feature importance plots
- Temporal transaction paths
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def visualize_explanation(
    explanation: Dict,
    node_labels: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a node explanation with subgraph and feature importance.

    Args:
        explanation: Explanation dictionary from GNNExplainer
        node_labels: Optional mapping of node indices to labels
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Subgraph visualization
    ax1 = axes[0]
    if HAS_NETWORKX:
        _plot_subgraph(explanation, ax1, node_labels)
    else:
        ax1.text(0.5, 0.5, "NetworkX not installed\nCannot visualize subgraph",
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Subgraph (requires NetworkX)")

    # Right: Feature importance
    ax2 = axes[1]
    _plot_feature_importance(explanation['feature_importance'], ax2)

    # Add prediction info
    pred = explanation['prediction']
    fig.suptitle(
        f"Node {explanation['node_idx']} - "
        f"Predicted: {'Illicit' if pred['class'] == 1 else 'Licit'} "
        f"(P={pred['probability_illicit']:.3f})",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _plot_subgraph(
    explanation: Dict,
    ax: plt.Axes,
    node_labels: Optional[Dict[int, str]] = None
):
    """Plot the explanation subgraph."""
    if not HAS_NETWORKX:
        return

    # Create graph from edges
    edges = explanation['subgraph_edges']
    G = nx.DiGraph()

    # Add nodes
    nodes = explanation['subgraph_nodes']
    for node in nodes:
        G.add_node(node)

    # Add edges with importance weights
    edge_importance = explanation['edge_importance']
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        # Map to original node indices
        src_orig = nodes[src]
        dst_orig = nodes[dst]
        weight = edge_importance[i] if i < len(edge_importance) else 0.1
        G.add_edge(src_orig, dst_orig, weight=weight)

    # Node colors based on whether it's the target
    target_node = explanation['node_idx']
    node_colors = ['red' if n == target_node else 'lightblue' for n in G.nodes()]

    # Edge colors based on importance
    if len(G.edges()) > 0:
        edge_weights = [G[u][v].get('weight', 0.1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_colors = [plt.cm.Reds(w / max_weight) for w in edge_weights]
        edge_widths = [1 + 3 * w / max_weight for w in edge_weights]
    else:
        edge_colors = []
        edge_widths = []

    # Layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except Exception:
        pos = nx.circular_layout(G)

    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=500, alpha=0.8)
    if len(G.edges()) > 0:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                              width=edge_widths, alpha=0.7,
                              arrows=True, arrowsize=15)

    # Labels
    if node_labels:
        labels = {n: node_labels.get(n, str(n)) for n in G.nodes()}
    else:
        labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    ax.set_title(f"Explanation Subgraph\n({len(nodes)} nodes, {edges.shape[1]} edges)")
    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(color='red', label='Target Node'),
        mpatches.Patch(color='lightblue', label='Neighbor Node'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')


def _plot_feature_importance(
    feature_importance: np.ndarray,
    ax: plt.Axes,
    top_k: int = 20
):
    """Plot top feature importances."""
    # Get top-k features
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    top_values = feature_importance[top_indices]

    # Normalize for visualization
    if top_values.max() > 0:
        top_values = top_values / top_values.max()

    # Plot horizontal bar chart
    y_pos = np.arange(len(top_indices))
    colors = plt.cm.Blues(0.3 + 0.7 * top_values)

    ax.barh(y_pos, top_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Feature {i}' for i in top_indices])
    ax.set_xlabel('Importance (normalized)')
    ax.set_title(f'Top {top_k} Feature Importances')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')


def visualize_subgraph(
    edge_index: np.ndarray,
    node_features: Optional[np.ndarray] = None,
    node_labels: Optional[np.ndarray] = None,
    highlight_nodes: Optional[List[int]] = None,
    title: str = "Transaction Subgraph",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a transaction subgraph.

    Args:
        edge_index: Edge connectivity (2, num_edges)
        node_features: Optional node features for sizing
        node_labels: Optional node labels (0=licit, 1=illicit)
        highlight_nodes: Nodes to highlight
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if not HAS_NETWORKX:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "NetworkX not installed",
               ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Create graph
    G = nx.DiGraph()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    # Node colors
    if node_labels is not None:
        color_map = {0: 'lightgreen', 1: 'red', -1: 'gray'}
        node_colors = [color_map.get(node_labels[n], 'gray') for n in G.nodes()]
    else:
        node_colors = 'lightblue'

    # Highlight specific nodes
    if highlight_nodes:
        node_colors = [
            'yellow' if n in highlight_nodes else c
            for n, c in zip(G.nodes(), node_colors if isinstance(node_colors, list) else [node_colors]*len(G.nodes()))
        ]

    # Node sizes
    if node_features is not None:
        # Use feature norm as size
        sizes = [300 + 200 * np.linalg.norm(node_features[n]) for n in G.nodes()]
    else:
        sizes = 500

    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5,
                          arrows=True, arrowsize=10)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    ax.set_title(title)
    ax.axis('off')

    # Legend
    if node_labels is not None:
        legend_elements = [
            mpatches.Patch(color='lightgreen', label='Licit'),
            mpatches.Patch(color='red', label='Illicit'),
            mpatches.Patch(color='gray', label='Unknown'),
        ]
        if highlight_nodes:
            legend_elements.append(mpatches.Patch(color='yellow', label='Highlighted'))
        ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature importance as a bar chart.

    Args:
        importance: Feature importance values
        feature_names: Optional feature names
        top_k: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get top-k features
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_values = importance[top_indices]

    # Feature names
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]

    top_names = [feature_names[i] for i in top_indices]

    # Plot
    y_pos = np.arange(len(top_indices))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_indices)))

    ax.barh(y_pos, top_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_temporal_pattern(
    timesteps: np.ndarray,
    values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Temporal Activity Pattern",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot temporal patterns of transactions.

    Args:
        timesteps: Timestep indices
        values: Values to plot (e.g., risk scores)
        labels: Optional labels for coloring
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        # Color by label
        colors = ['red' if l == 1 else 'green' if l == 0 else 'gray' for l in labels]
        ax.scatter(timesteps, values, c=colors, alpha=0.6, s=20)

        # Legend
        legend_elements = [
            mpatches.Patch(color='red', label='Illicit'),
            mpatches.Patch(color='green', label='Licit'),
            mpatches.Patch(color='gray', label='Unknown'),
        ]
        ax.legend(handles=legend_elements)
    else:
        ax.scatter(timesteps, values, alpha=0.6, s=20)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_explanation_report(
    explanations: List[Dict],
    save_dir: str = "explanations",
) -> str:
    """
    Create an HTML report of multiple explanations.

    Args:
        explanations: List of explanation dictionaries
        save_dir: Directory to save report

    Returns:
        Path to generated report
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GNN Explanation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .explanation { border: 1px solid #ccc; padding: 15px; margin: 10px 0; }
            .illicit { border-color: red; background-color: #ffeeee; }
            .licit { border-color: green; background-color: #eeffee; }
            h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f4f4f4; }
        </style>
    </head>
    <body>
        <h1>GNN Explanation Report</h1>
        <p>Generated explanations for high-risk transactions.</p>
    """

    for i, exp in enumerate(explanations):
        pred = exp['prediction']
        css_class = 'illicit' if pred['class'] == 1 else 'licit'

        html_content += f"""
        <div class="explanation {css_class}">
            <h2>Node {exp['node_idx']}</h2>
            <p><strong>Prediction:</strong> {'Illicit' if pred['class'] == 1 else 'Licit'}</p>
            <p><strong>Probability (Illicit):</strong> {pred['probability_illicit']:.4f}</p>
            <p><strong>Subgraph Size:</strong> {len(exp['subgraph_nodes'])} nodes</p>

            <h3>Top Important Edges</h3>
            <table>
                <tr><th>Source</th><th>Target</th><th>Importance</th></tr>
        """

        for j in range(min(5, exp['top_edges'].shape[1])):
            src = exp['top_edges'][0, j]
            dst = exp['top_edges'][1, j]
            score = exp['top_edge_scores'][j]
            html_content += f"<tr><td>{src}</td><td>{dst}</td><td>{score:.4f}</td></tr>"

        html_content += """
            </table>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    report_path = os.path.join(save_dir, "explanation_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization module...")

    # Create dummy explanation
    explanation = {
        'node_idx': 42,
        'subgraph_nodes': np.array([42, 10, 20, 30, 40, 50]),
        'subgraph_edges': np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        'feature_importance': np.random.rand(166),
        'edge_importance': np.random.rand(5),
        'top_edges': np.array([[42, 10], [10, 20]]).T,
        'top_edge_scores': np.array([0.8, 0.5]),
        'prediction': {
            'class': 1,
            'probability_illicit': 0.85,
            'probability_licit': 0.15,
        }
    }

    # Test visualization
    fig = visualize_explanation(explanation)
    plt.close(fig)

    print("Visualization test passed!")
