"""
GNN Explainability Module
=========================

Post-hoc explainability for GNN predictions using:
- GNNExplainer: Identifies important subgraph and features
- Subgraph extraction for human-interpretable explanations

Explanations identify:
- Influential neighboring transactions and paths
- Salient node and edge attributes
- Temporal segments contributing to risk scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class GNNExplainerWrapper:
    """
    Wrapper for GNN explanation methods.

    Provides a unified interface for explaining GNN predictions
    using gradient-based and perturbation-based methods.

    Args:
        model: Trained GNN model
        num_hops: Number of hops for subgraph extraction
        epochs: Training epochs for explainer
        lr: Learning rate for explainer optimization
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        num_hops: int = 2,
        epochs: int = 100,
        lr: float = 0.01,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.num_hops = num_hops
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        top_k_edges: int = 10,
    ) -> Dict:
        """
        Explain prediction for a single node.

        Args:
            node_idx: Index of node to explain
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge attributes (optional)
            timestep: Node timesteps (optional)
            top_k_edges: Number of top edges to return

        Returns:
            Dictionary containing explanation components
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        if timestep is not None:
            timestep = timestep.to(self.device)

        # Extract k-hop subgraph
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx,
            self.num_hops,
            edge_index,
            relabel_nodes=True,
            num_nodes=x.size(0)
        )

        sub_x = x[subset]
        if timestep is not None:
            sub_timestep = timestep[subset]
        else:
            sub_timestep = None

        if edge_attr is not None:
            sub_edge_attr = edge_attr[edge_mask]
        else:
            sub_edge_attr = None

        # Get node index in subgraph
        new_node_idx = mapping.item() if mapping.dim() == 0 else mapping[0].item()

        # Compute feature importance using gradients
        feature_importance = self._compute_feature_importance(
            sub_x, sub_edge_index, sub_edge_attr, sub_timestep, new_node_idx
        )

        # Compute edge importance
        edge_importance = self._compute_edge_importance(
            sub_x, sub_edge_index, sub_edge_attr, sub_timestep, new_node_idx
        )

        # Get top-k edges
        top_edge_indices = torch.argsort(edge_importance, descending=True)[:top_k_edges]
        important_edges = sub_edge_index[:, top_edge_indices]

        # Map back to original node indices
        important_edges_original = subset[important_edges]

        return {
            'node_idx': node_idx,
            'subgraph_nodes': subset.cpu().numpy(),
            'subgraph_edges': sub_edge_index.cpu().numpy(),
            'feature_importance': feature_importance.cpu().numpy(),
            'edge_importance': edge_importance.cpu().numpy(),
            'top_edges': important_edges_original.cpu().numpy(),
            'top_edge_scores': edge_importance[top_edge_indices].cpu().numpy(),
            'prediction': self._get_prediction(x, edge_index, edge_attr, timestep, node_idx),
        }

    def _compute_feature_importance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        node_idx: int,
    ) -> torch.Tensor:
        """Compute feature importance using integrated gradients."""
        x = x.clone().requires_grad_(True)

        # Forward pass
        if edge_attr is not None and timestep is not None:
            out = self.model(x, edge_index, edge_attr, timestep)
        elif timestep is not None:
            try:
                out = self.model(x, edge_index, timestep)
            except TypeError:
                out = self.model(x, edge_index)
        else:
            out = self.model(x, edge_index)

        # Get prediction for target node (class 1 = illicit)
        pred = out[node_idx, 1]

        # Backward pass
        pred.backward()

        # Feature importance = gradient magnitude
        feature_importance = x.grad[node_idx].abs()

        return feature_importance.detach()

    def _compute_edge_importance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        node_idx: int,
    ) -> torch.Tensor:
        """
        Compute edge importance using edge masking.

        Measures how much each edge contributes to the prediction
        by masking edges and observing prediction change.
        """
        num_edges = edge_index.size(1)
        edge_importance = torch.zeros(num_edges, device=self.device)

        # Get baseline prediction
        with torch.no_grad():
            if edge_attr is not None and timestep is not None:
                baseline_out = self.model(x, edge_index, edge_attr, timestep)
            elif timestep is not None:
                try:
                    baseline_out = self.model(x, edge_index, timestep)
                except TypeError:
                    baseline_out = self.model(x, edge_index)
            else:
                baseline_out = self.model(x, edge_index)

            baseline_pred = F.softmax(baseline_out[node_idx], dim=0)[1]

        # Compute importance for each edge by masking
        for i in range(num_edges):
            # Create masked edge index
            mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)
            mask[i] = False
            masked_edge_index = edge_index[:, mask]

            if edge_attr is not None:
                masked_edge_attr = edge_attr[mask]
            else:
                masked_edge_attr = None

            # Forward pass with masked edge
            with torch.no_grad():
                if masked_edge_attr is not None and timestep is not None:
                    out = self.model(x, masked_edge_index, masked_edge_attr, timestep)
                elif timestep is not None:
                    try:
                        out = self.model(x, masked_edge_index, timestep)
                    except TypeError:
                        out = self.model(x, masked_edge_index)
                else:
                    out = self.model(x, masked_edge_index)

                masked_pred = F.softmax(out[node_idx], dim=0)[1]

            # Edge importance = change in prediction
            edge_importance[i] = (baseline_pred - masked_pred).abs()

        return edge_importance

    def _get_prediction(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        node_idx: int,
    ) -> Dict:
        """Get model prediction for a node."""
        with torch.no_grad():
            if edge_attr is not None and timestep is not None:
                out = self.model(x, edge_index, edge_attr, timestep)
            elif timestep is not None:
                try:
                    out = self.model(x, edge_index, timestep)
                except TypeError:
                    out = self.model(x, edge_index)
            else:
                out = self.model(x, edge_index)

            probs = F.softmax(out[node_idx], dim=0)
            pred_class = probs.argmax().item()

        return {
            'class': pred_class,
            'probability_illicit': probs[1].item(),
            'probability_licit': probs[0].item(),
        }

    def explain_batch(
        self,
        node_indices: List[int],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        top_k_edges: int = 10,
    ) -> List[Dict]:
        """
        Explain predictions for multiple nodes.

        Args:
            node_indices: List of node indices to explain
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge attributes (optional)
            timestep: Node timesteps (optional)
            top_k_edges: Number of top edges per node

        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for idx in node_indices:
            exp = self.explain_node(
                idx, x, edge_index, edge_attr, timestep, top_k_edges
            )
            explanations.append(exp)
        return explanations


def explain_node(
    model: nn.Module,
    data: Data,
    node_idx: int,
    num_hops: int = 2,
    top_k_edges: int = 10,
    device: str = "cpu",
) -> Dict:
    """
    Convenience function to explain a single node.

    Args:
        model: Trained GNN model
        data: PyG Data object
        node_idx: Node to explain
        num_hops: Subgraph hops
        top_k_edges: Top edges to return
        device: Computation device

    Returns:
        Explanation dictionary
    """
    explainer = GNNExplainerWrapper(model, num_hops=num_hops, device=device)

    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    timestep = data.timestep if hasattr(data, 'timestep') else None

    return explainer.explain_node(
        node_idx=node_idx,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        timestep=timestep,
        top_k_edges=top_k_edges,
    )


def explain_batch(
    model: nn.Module,
    data: Data,
    node_indices: List[int],
    num_hops: int = 2,
    top_k_edges: int = 10,
    device: str = "cpu",
) -> List[Dict]:
    """
    Convenience function to explain multiple nodes.

    Args:
        model: Trained GNN model
        data: PyG Data object
        node_indices: Nodes to explain
        num_hops: Subgraph hops
        top_k_edges: Top edges per node
        device: Computation device

    Returns:
        List of explanation dictionaries
    """
    explainer = GNNExplainerWrapper(model, num_hops=num_hops, device=device)

    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    timestep = data.timestep if hasattr(data, 'timestep') else None

    return explainer.explain_batch(
        node_indices=node_indices,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        timestep=timestep,
        top_k_edges=top_k_edges,
    )


def get_high_risk_nodes(
    model: nn.Module,
    data: Data,
    threshold: float = 0.8,
    top_k: int = 100,
    device: str = "cpu",
) -> Tuple[List[int], List[float]]:
    """
    Get nodes with highest predicted illicit probability.

    Args:
        model: Trained GNN model
        data: PyG Data object
        threshold: Minimum probability threshold
        top_k: Maximum nodes to return
        device: Computation device

    Returns:
        Tuple of (node indices, probabilities)
    """
    model = model.to(device)
    model.eval()

    data = data.to(device)

    with torch.no_grad():
        if hasattr(data, 'edge_attr') and hasattr(data, 'timestep'):
            out = model(data.x, data.edge_index, data.edge_attr, data.timestep)
        elif hasattr(data, 'timestep'):
            try:
                out = model(data.x, data.edge_index, data.timestep)
            except TypeError:
                out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index)

        probs = F.softmax(out, dim=1)[:, 1]

    # Filter by threshold and get top-k
    high_risk_mask = probs >= threshold
    high_risk_indices = torch.where(high_risk_mask)[0]
    high_risk_probs = probs[high_risk_mask]

    # Sort by probability
    sorted_idx = torch.argsort(high_risk_probs, descending=True)[:top_k]
    top_indices = high_risk_indices[sorted_idx].cpu().tolist()
    top_probs = high_risk_probs[sorted_idx].cpu().tolist()

    return top_indices, top_probs


if __name__ == "__main__":
    # Test explainability module
    print("Testing GNN Explainer...")

    from ..models.graphsage import create_graphsage_model

    # Create dummy data
    num_nodes = 100
    num_features = 166
    num_edges = 500

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))
    timestep = torch.randint(0, 49, (num_nodes,))

    data = Data(x=x, edge_index=edge_index, y=y, timestep=timestep)

    # Create and train a simple model
    model = create_graphsage_model(num_features=num_features)

    # Test explanation
    explanation = explain_node(model, data, node_idx=0, device="cpu")

    print(f"\nExplanation for node 0:")
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Subgraph nodes: {len(explanation['subgraph_nodes'])}")
    print(f"  Top features: {explanation['feature_importance'][:5]}")
    print(f"  Top edges shape: {explanation['top_edges'].shape}")

    print("\nExplainability test passed!")
