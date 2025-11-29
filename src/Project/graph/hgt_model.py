"""Heterogeneous Graph Transformer (HGT) model for evidence binding."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGT(nn.Module):
    """Heterogeneous Graph Transformer for post-level criteria prediction.

    Uses HGTConv layers to aggregate information across heterogeneous graph structure,
    then predicts both edge-level (sentence-criterion) and node-level (post-criterion)
    relationships.

    Args:
        metadata: PyG metadata tuple containing node types and edge types
        embedding_dim: Dimension of input node embeddings (from BGE-M3)
        hidden_channels: Hidden dimension for HGT layers
        out_channels: Output dimension
        num_heads: Number of attention heads
        num_layers: Number of HGT layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        metadata: Tuple,
        embedding_dim: int = 1024,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # Input projection layers for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(embedding_dim, hidden_channels)

        # HGT convolutional layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # Output layers
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 5, hidden_channels),  # 5 for edge features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1),
        )

        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # post + criterion
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1),
        )

        self.dropout = dropout

    def forward(self, data):
        """Forward pass.

        Args:
            data: HeteroData batch

        Returns:
            edge_logits: Sentence-criterion edge predictions
            node_logits: Post-criterion node predictions
        """
        # Project input embeddings to hidden dimension
        x_dict = {}
        for node_type, x in data.x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x)
            x_dict[node_type] = F.relu(x_dict[node_type])
            x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)

        # Apply HGT convolutional layers
        # Filter edge_index_dict to only include edges for message passing
        # Exclude structural edges like ('post', 'contains', 'sentence')
        mp_edge_index_dict = {
            k: v for k, v in data.edge_index_dict.items()
            if k in [('sentence', 'supports', 'criterion'), ('sentence', 'next', 'sentence')]
        }

        for conv in self.convs:
            if mp_edge_index_dict:  # Only apply if we have edges to process
                # Store post embeddings before message passing
                post_emb = x_dict.get("post", None)

                # Apply message passing (only updates nodes with edges)
                x_dict = conv(x_dict, mp_edge_index_dict)

                # Restore post embeddings if they were removed
                if post_emb is not None and "post" not in x_dict:
                    x_dict["post"] = post_emb

                # Apply activation and dropout
                x_dict = {
                    key: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }
            else:
                # No message passing, just apply activation and dropout
                x_dict = {
                    key: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }

        # Edge-level prediction (sentence -> criterion)
        edge_logits = self._predict_edges(data, x_dict)

        # Node-level prediction (post -> criterion)
        node_logits = self._predict_nodes(data, x_dict)

        return edge_logits, node_logits

    def _predict_edges(self, data, x_dict):
        """Predict sentence-criterion edges."""
        if ("sentence", "supports", "criterion") not in data.edge_index_dict:
            return torch.tensor([], device=next(self.parameters()).device)

        edge_index = data.edge_index_dict[("sentence", "supports", "criterion")]
        edge_attr = data.edge_attr_dict.get(("sentence", "supports", "criterion"), None)

        # Get sentence and criterion embeddings
        sent_emb = x_dict["sentence"][edge_index[0]]  # Source nodes (sentences)
        crit_emb = x_dict["criterion"][edge_index[1]]  # Target nodes (criteria)

        # Concatenate node embeddings and edge features
        if edge_attr is not None:
            edge_input = torch.cat([sent_emb, crit_emb, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([sent_emb, crit_emb], dim=-1)

        # Predict edge logits
        edge_logits = self.edge_predictor(edge_input).squeeze(-1)

        return edge_logits

    def _predict_nodes(self, data, x_dict):
        """Predict post-criterion relationships."""
        if ("post", "matches", "criterion") not in data.edge_index_dict:
            # Fallback: create edges for all (post, criterion) pairs within each graph
            # Use batch information to only pair posts with criteria from same graph
            post_emb = x_dict["post"]  # Shape: [num_posts_in_batch, hidden]
            crit_emb = x_dict["criterion"]  # Shape: [num_criteria_in_batch, hidden]

            # Get batch indices
            post_batch = data["post"].batch if hasattr(data["post"], "batch") else torch.zeros(post_emb.size(0), dtype=torch.long, device=post_emb.device)
            crit_batch = data["criterion"].batch if hasattr(data["criterion"], "batch") else torch.zeros(crit_emb.size(0), dtype=torch.long, device=crit_emb.device)

            # Build edges: for each post, connect to all criteria in same graph
            edge_src = []
            edge_dst = []
            for post_idx in range(post_emb.size(0)):
                post_graph = post_batch[post_idx]
                # Find all criteria in same graph
                crit_mask = (crit_batch == post_graph)
                crit_indices = torch.where(crit_mask)[0]
                # Add edges
                edge_src.extend([post_idx] * len(crit_indices))
                edge_dst.extend(crit_indices.tolist())

            edge_src = torch.tensor(edge_src, dtype=torch.long, device=post_emb.device)
            edge_dst = torch.tensor(edge_dst, dtype=torch.long, device=post_emb.device)

            # Get embeddings
            post_emb_selected = post_emb[edge_src]
            crit_emb_selected = crit_emb[edge_dst]

            node_input = torch.cat([post_emb_selected, crit_emb_selected], dim=-1)
            node_logits = self.node_predictor(node_input).squeeze(-1)
        else:
            edge_index = data.edge_index_dict[("post", "matches", "criterion")]
            post_emb = x_dict["post"][edge_index[0]]
            crit_emb = x_dict["criterion"][edge_index[1]]

            node_input = torch.cat([post_emb, crit_emb], dim=-1)
            node_logits = self.node_predictor(node_input).squeeze(-1)

        return node_logits


def create_hgt_model(
    metadata: Tuple,
    cfg: Dict,
    embedding_dim: int = 1024,
) -> HGT:
    """Factory function to create HGT model from config.

    Args:
        metadata: PyG metadata tuple
        cfg: Configuration dict with model parameters
        embedding_dim: Dimension of input embeddings (from BGE-M3)

    Returns:
        HGT model instance
    """
    model_cfg = cfg.get("model", {})

    return HGT(
        metadata=metadata,
        embedding_dim=embedding_dim,
        hidden_channels=model_cfg.get("hidden", 256),
        out_channels=model_cfg.get("out_channels", 128),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("layers", 2),
        dropout=model_cfg.get("dropout", 0.2),
    )
