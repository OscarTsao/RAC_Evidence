"""Training loop for heterogeneous GNN refinement."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch import amp
from omegaconf import DictConfig

from Project.graph.hetero_gnn import SimpleHeteroGNN
from Project.utils import enable_performance_optimizations
from Project.utils.hydra_utils import load_config
from Project.utils.logging import get_logger


def _consistency_regularizer(
    edge_logits: torch.Tensor,
    edge_index: torch.Tensor,
    node_logits: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    if edge_logits.numel() == 0:
        return torch.tensor(0.0, device=node_logits.device)
    probs_edge = torch.sigmoid(edge_logits)
    node_probs = torch.sigmoid(node_logits)
    total = torch.tensor(0.0, device=node_logits.device)
    for crit_idx in edge_index[1].unique():
        mask = edge_index[1] == crit_idx
        max_edge = probs_edge[mask].max()
        node_prob = node_probs[crit_idx]
        total = total + torch.relu(margin + max_edge - node_prob) + torch.relu(
            margin + node_prob - max_edge
        )
    return total


def train_gnn(cfg: DictConfig) -> Path:
    # Enable PyTorch performance optimizations
    enable_performance_optimizations()

    logger = get_logger(__name__)
    processed_dir = Path(cfg.data.processed_dir)
    graphs_path = processed_dir / "graphs.pt"
    graphs: List = torch.load(graphs_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleHeteroGNN(
        hidden=cfg.model.hidden,
        dropout=cfg.model.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    amp_dtype = str(
        cfg.train.get("amp_dtype", "bf16" if cfg.train.get("fp16", False) else "fp32")
    ).lower()
    use_amp = (
        bool(cfg.train.get("use_amp", cfg.train.get("fp16", False) or amp_dtype in ("bf16", "fp16")))
        and device.type == "cuda"
    )
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = amp.GradScaler("cuda", enabled=use_amp and amp_dtype == "fp16")

    model.train()
    for _ in range(cfg.train.epochs):
        for graph in graphs:
            graph = graph.to(device)

            opt.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Mixed precision forward pass
            with amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype if use_amp else None):
                edge_logits, node_logits = model(graph)
                loss_edge = torch.tensor(0.0, device=device)
                loss_node = torch.tensor(0.0, device=device)
                if ("sentence", "supports", "criterion") in graph:
                    labels = graph[("sentence", "supports", "criterion")].get(
                        "edge_label", torch.zeros_like(edge_logits)
                    )
                    loss_edge = bce(edge_logits, labels)
                    edge_index = graph[("sentence", "supports", "criterion")]["edge_index"]
                else:
                    edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
                criterion_labels = graph["criterion"].get("y", torch.zeros_like(node_logits))
                if criterion_labels.numel() > 0:
                    loss_node = bce(node_logits, criterion_labels)
                cons_loss = _consistency_regularizer(edge_logits, edge_index, node_logits, cfg.loss.cons_margin)
                loss = (
                    cfg.loss.lambda_edge * loss_edge
                    + cfg.loss.lambda_node * loss_node
                    + cfg.loss.lambda_cons * cons_loss
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
    exp = cfg.get("exp", "debug")
    out_dir = Path(cfg.get("output", f"outputs/runs/{exp}")) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "gnn.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info("Saved GNN checkpoint to %s", ckpt_path)
    return ckpt_path


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    train_gnn(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
