"""Heterogeneous GNN for refinement (lightweight fallback)."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from Project.graph.common import HeteroData


class SimpleHeteroGNN(nn.Module):
    """Small MLP-based network that consumes edge/node attributes."""

    def __init__(self, edge_feat_dim: int = 6, node_feat_dim: int = 2, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        sc_store = data.get(("sentence", "supports", "criterion"), {})
        edge_attr = sc_store.get("edge_attr", torch.zeros((0, 6)))
        edge_logits = self.edge_mlp(edge_attr).squeeze(-1)
        criterion_x = data["criterion"].get("x", torch.zeros((0, 2)))
        node_logits = self.node_mlp(criterion_x).squeeze(-1)
        return edge_logits, node_logits
