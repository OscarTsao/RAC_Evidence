"""Retrieval recall evaluation.

Evaluates how well the retrieval stage (before reranking) captures positive evidence.
Computes retrieval recall@K for different K values and per-criterion.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from Project.utils.logging import get_logger


def normalize_key(value):
    """Normalize ID values to strings for consistent comparison."""
    return str(value)


def compute_retrieval_recall(
    fold_predictions: List[Dict],
    gold_labels: List[Dict] | None = None,
) -> Dict:
    """Compute retrieval recall metrics.

    Retrieval recall@K measures: "Of all positive evidence sentences,
    what fraction were retrieved in the top-K candidates?"

    Args:
        fold_predictions: List of predictions from fold dumps
            Each prediction has: post_id, sent_id, cid, label, prob, etc.
        gold_labels: Optional list of gold labels. If None, uses label field
            from predictions.

    Returns:
        Dict with overall and per-criterion retrieval recall metrics
    """
    logger = get_logger(__name__)

    # Extract retrieved (post, sent, cid) triples
    retrieved_set = set()
    for pred in fold_predictions:
        key = (normalize_key(pred["post_id"]), normalize_key(pred["sent_id"]), normalize_key(pred["cid"]))
        retrieved_set.add(key)

    # Extract gold positives
    if gold_labels is None:
        # Use labels from predictions
        gold_positives = set()
        for pred in fold_predictions:
            if pred["label"] == 1:
                key = (normalize_key(pred["post_id"]), normalize_key(pred["sent_id"]), normalize_key(pred["cid"]))
                gold_positives.add(key)
    else:
        # Use external gold labels
        gold_positives = set()
        for label in gold_labels:
            if label.get("label") == 1 or label.get("groundtruth") == 1:
                key = (normalize_key(label["post_id"]), normalize_key(label["sent_id"]), normalize_key(label["cid"]))
                gold_positives.add(key)

    # Compute overall recall
    retrieved_positives = retrieved_set & gold_positives
    overall_recall = len(retrieved_positives) / len(gold_positives) if gold_positives else 0.0

    logger.info(f"Retrieval recall: {len(retrieved_positives)}/{len(gold_positives)} = {overall_recall:.4f}")

    # Per-criterion recall
    by_crit: Dict[str, Dict[str, set]] = defaultdict(lambda: {"retrieved": set(), "gold": set()})

    for pred in fold_predictions:
        cid = normalize_key(pred["cid"])
        key = (normalize_key(pred["post_id"]), normalize_key(pred["sent_id"]))
        by_crit[cid]["retrieved"].add(key)

        if pred["label"] == 1:
            by_crit[cid]["gold"].add(key)

    per_criterion = {}
    for cid, data in by_crit.items():
        retrieved = data["retrieved"]
        gold = data["gold"]
        retrieved_pos = retrieved & gold

        recall = len(retrieved_pos) / len(gold) if gold else 0.0
        per_criterion[cid] = {
            "n_retrieved": len(retrieved),
            "n_gold_positives": len(gold),
            "n_retrieved_positives": len(retrieved_pos),
            "retrieval_recall": float(recall),
        }

    metrics = {
        "overall": {
            "n_retrieved": len(retrieved_set),
            "n_gold_positives": len(gold_positives),
            "n_retrieved_positives": len(retrieved_positives),
            "retrieval_recall": float(overall_recall),
        },
        "per_criterion": per_criterion,
    }

    return metrics


def compute_retrieval_recall_at_k(
    fold_predictions: List[Dict],
    k_values: List[int] = [5, 10, 20],
    prob_key: str = "prob",
) -> Dict:
    """Compute retrieval recall@K for different K values.

    For each (post_id, cid) group, takes top-K by probability and checks
    if gold positives appear in that top-K.

    Args:
        fold_predictions: List of predictions
        k_values: List of K values to evaluate
        prob_key: Key to use for ranking

    Returns:
        Dict with recall@K metrics
    """
    logger = get_logger(__name__)

    # Group by (post_id, cid)
    by_group = defaultdict(list)
    for pred in fold_predictions:
        key = (normalize_key(pred["post_id"]), normalize_key(pred["cid"]))
        by_group[key].append(pred)

    # Compute recall@K for each group
    metrics = {"k_values": k_values}

    for k in k_values:
        total_positives = 0
        retrieved_positives = 0

        for group_preds in by_group.values():
            # Sort by score descending
            sorted_preds = sorted(group_preds, key=lambda x: x.get(prob_key, 0.0), reverse=True)
            topk = sorted_preds[:k]

            # Count gold positives
            group_gold = [p for p in group_preds if p["label"] == 1]
            topk_gold = [p for p in topk if p["label"] == 1]

            total_positives += len(group_gold)
            retrieved_positives += len(topk_gold)

        recall = retrieved_positives / total_positives if total_positives > 0 else 0.0
        metrics[f"recall@{k}"] = float(recall)
        logger.info(f"Retrieval Recall@{k}: {retrieved_positives}/{total_positives} = {recall:.4f}")

    metrics["total_positives"] = total_positives

    return metrics


def evaluate_fold_retrieval(
    fold_dir: Path,
    n_folds: int = 5,
    k_values: List[int] = [5, 10, 20],
) -> Dict:
    """Evaluate retrieval recall across all folds.

    Args:
        fold_dir: Directory containing fold_{i}_predictions.jsonl files
        n_folds: Number of folds
        k_values: K values for recall@K evaluation

    Returns:
        Dict with aggregated retrieval metrics
    """
    logger = get_logger(__name__)

    all_predictions = []

    for fold in range(n_folds):
        fold_pred_path = fold_dir / f"fold_{fold}_predictions.jsonl"
        if not fold_pred_path.exists():
            logger.warning(f"Fold {fold} predictions not found: {fold_pred_path}")
            continue

        with open(fold_pred_path) as f:
            fold_preds = [json.loads(line) for line in f]
            all_predictions.extend(fold_preds)
            logger.info(f"Loaded {len(fold_preds)} predictions from fold {fold}")

    if not all_predictions:
        raise ValueError(f"No fold predictions found in {fold_dir}")

    # Compute overall retrieval recall
    recall_metrics = compute_retrieval_recall(all_predictions)

    # Compute recall@K
    recall_at_k = compute_retrieval_recall_at_k(all_predictions, k_values)

    # Combine metrics
    metrics = {
        "n_folds": n_folds,
        "n_predictions": len(all_predictions),
        "retrieval_recall": recall_metrics,
        "retrieval_recall_at_k": recall_at_k,
    }

    return metrics


def save_retrieval_metrics(
    fold_dir: Path,
    output_path: Path,
    n_folds: int = 5,
    k_values: List[int] = [5, 10, 20],
) -> None:
    """Evaluate and save retrieval metrics.

    Args:
        fold_dir: Directory with fold predictions
        output_path: Path to save metrics JSON
        n_folds: Number of folds
        k_values: K values for evaluation
    """
    logger = get_logger(__name__)

    metrics = evaluate_fold_retrieval(fold_dir, n_folds, k_values)

    # Save metrics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved retrieval metrics to {output_path}")

    # Print summary
    overall_recall = metrics["retrieval_recall"]["overall"]["retrieval_recall"]
    logger.info(f"Overall retrieval recall: {overall_recall:.4f}")

    for k in k_values:
        recall_k = metrics["retrieval_recall_at_k"][f"recall@{k}"]
        logger.info(f"Retrieval Recall@{k}: {recall_k:.4f}")
