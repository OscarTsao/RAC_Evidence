"""Training script for HGT model on heterogeneous graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

from Project.graph.dataset import load_hetero_dataset
from Project.graph.hgt_model import create_hgt_model
from Project.utils import enable_performance_optimizations
from Project.utils.hydra_utils import load_config
from Project.utils.logging import get_logger


def consistency_loss(
    edge_logits: torch.Tensor,
    edge_index: torch.Tensor,
    node_logits: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Compute consistency loss between edge-level and node-level predictions.

    Encourages that max(edge_prob) for a criterion is close to node_prob for that criterion.

    Args:
        edge_logits: Sentence-criterion edge predictions
        edge_index: Edge indices [2, num_edges]
        node_logits: Post-criterion node predictions
        margin: Margin for consistency constraint

    Returns:
        Consistency loss value
    """
    if edge_logits.numel() == 0:
        return torch.tensor(0.0, device=node_logits.device)

    edge_probs = torch.sigmoid(edge_logits)
    node_probs = torch.sigmoid(node_logits)

    total_loss = torch.tensor(0.0, device=node_logits.device)

    # For each criterion, ensure max edge probability is consistent with node probability
    for crit_idx in edge_index[1].unique():
        mask = edge_index[1] == crit_idx
        max_edge_prob = edge_probs[mask].max()
        node_prob = node_probs[crit_idx]

        # Bidirectional consistency: both should be similar
        total_loss = total_loss + torch.relu(margin + max_edge_prob - node_prob)
        total_loss = total_loss + torch.relu(margin + node_prob - max_edge_prob)

    return total_loss / edge_index[1].unique().numel()


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        preds: Binary predictions
        labels: Ground truth labels
        probs: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    probs_np = probs.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average='binary', zero_division=0
    )
    macro_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = 0.0

    return {
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: HGT model
        loader: DataLoader for training graphs
        optimizer: Optimizer
        cfg: Configuration
        device: Device to use
        scaler: GradScaler for mixed precision
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    total_edge_loss = 0.0
    total_node_loss = 0.0
    total_cons_loss = 0.0
    num_batches = 0

    bce_loss = nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            edge_logits, node_logits = model(batch)

            # Edge loss
            loss_edge = torch.tensor(0.0, device=device)
            if ("sentence", "supports", "criterion") in batch.edge_index_dict:
                edge_labels = batch[("sentence", "supports", "criterion")].edge_label
                loss_edge = bce_loss(edge_logits, edge_labels)

            # Node loss
            loss_node = torch.tensor(0.0, device=device)
            if batch["criterion"].y is not None:
                node_labels = batch["criterion"].y
                loss_node = bce_loss(node_logits, node_labels)

            # Consistency loss
            edge_index = batch.edge_index_dict.get(("sentence", "supports", "criterion"), None)
            if edge_index is not None:
                loss_cons = consistency_loss(edge_logits, edge_index, node_logits, cfg.loss.cons_margin)
            else:
                loss_cons = torch.tensor(0.0, device=device)

            # Total loss
            loss = (
                cfg.loss.lambda_edge * loss_edge
                + cfg.loss.lambda_node * loss_node
                + cfg.loss.lambda_cons * loss_cons
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.get("grad_clip", 1.0))
            optimizer.step()

        total_loss += loss.item()
        total_edge_loss += loss_edge.item()
        total_node_loss += loss_node.item()
        total_cons_loss += loss_cons.item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "edge_loss": total_edge_loss / max(num_batches, 1),
        "node_loss": total_node_loss / max(num_batches, 1),
        "cons_loss": total_cons_loss / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation/test set.

    Args:
        model: HGT model
        loader: DataLoader for evaluation graphs
        device: Device to use

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_node_preds = []
    all_node_labels = []
    all_node_probs = []

    for batch in loader:
        batch = batch.to(device)
        _, node_logits = model(batch)

        node_probs = torch.sigmoid(node_logits)
        node_preds = (node_probs >= 0.5).float()

        all_node_preds.append(node_preds)
        all_node_labels.append(batch["criterion"].y)
        all_node_probs.append(node_probs)

    all_node_preds = torch.cat(all_node_preds)
    all_node_labels = torch.cat(all_node_labels)
    all_node_probs = torch.cat(all_node_probs)

    metrics = compute_metrics(all_node_preds, all_node_labels, all_node_probs)

    return metrics


def train_hgt(cfg: DictConfig) -> Path:
    """Main training function for HGT model.

    Args:
        cfg: Configuration

    Returns:
        Path to saved model checkpoint
    """
    enable_performance_optimizations()
    logger = get_logger(__name__)

    # Load dataset
    run_dir = Path(cfg.get("output_dir", f"outputs/runs/{cfg.exp}"))
    graphs_dir = run_dir / "graphs"

    logger.info(f"Loading dataset from {graphs_dir}")
    dataset = load_hetero_dataset(graphs_dir)
    logger.info(f"Loaded {len(dataset)} graphs")

    # Get metadata from first graph
    sample_graph = dataset[0]
    metadata = (list(sample_graph.node_types), list(sample_graph.edge_types))
    embedding_dim = sample_graph["sentence"].x.shape[1]

    logger.info(f"Graph metadata: {metadata}")
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Create data loaders
    batch_size = cfg.train.get("batch_size_posts", 4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.dataloader.get("num_workers", 2))

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_hgt_model(metadata, cfg, embedding_dim=embedding_dim)
    model = model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.get("weight_decay", 5e-4),
    )

    # Mixed precision setup
    use_amp = cfg.train.get("use_amp", True) and device.type == "cuda"
    amp_dtype = cfg.train.get("amp_dtype", "bf16")
    scaler = GradScaler(enabled=use_amp and amp_dtype == "fp16")

    logger.info(f"Training on {device} with AMP={use_amp} (dtype={amp_dtype})")

    # Training loop
    best_f1 = 0.0
    patience = cfg.train.get("patience", 10)
    patience_counter = 0

    for epoch in range(cfg.train.epochs):
        train_metrics = train_one_epoch(model, loader, optimizer, cfg, device, scaler, use_amp)

        logger.info(
            f"Epoch {epoch + 1}/{cfg.train.epochs} - "
            f"Loss: {train_metrics['loss']:.4f} "
            f"(Edge: {train_metrics['edge_loss']:.4f}, "
            f"Node: {train_metrics['node_loss']:.4f}, "
            f"Cons: {train_metrics['cons_loss']:.4f})"
        )

        # Evaluate
        if (epoch + 1) % cfg.train.get("eval_every", 1) == 0:
            eval_metrics = evaluate(model, loader, device)
            logger.info(
                f"Eval - F1: {eval_metrics['f1']:.4f}, "
                f"Macro-F1: {eval_metrics['macro_f1']:.4f}, "
                f"Precision: {eval_metrics['precision']:.4f}, "
                f"Recall: {eval_metrics['recall']:.4f}, "
                f"AUC: {eval_metrics['auc']:.4f}"
            )

            # Early stopping
            if eval_metrics['macro_f1'] > best_f1:
                best_f1 = eval_metrics['macro_f1']
                patience_counter = 0

                # Save best model
                ckpt_dir = run_dir / "gnn"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / "best_model.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "metrics": eval_metrics,
                }, ckpt_path)
                logger.info(f"Saved best model to {ckpt_path} (Macro-F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    # Save final metrics
    metrics_path = run_dir / "gnn" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "best_macro_f1": best_f1,
            "final_epoch": epoch + 1,
        }, f, indent=2)

    logger.info(f"Training complete. Best Macro-F1: {best_f1:.4f}")

    return run_dir / "gnn" / "best_model.pt"


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    train_hgt(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.cfg)
