"""Evidence-specific evaluation metrics.

Computes:
- AUPRC, F1, Precision@K, Coverage@K for sentence-criterion matching
- Recall@K for retrieval quality
- ECE (Expected Calibration Error) before/after temperature scaling
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from Project.metrics.calibration import ece as compute_ece


def precision_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int = 5
) -> float:
    """Compute Precision@K.

    Args:
        y_true: Ground truth labels
        y_score: Prediction scores
        k: Top-K

    Returns:
        Precision@K
    """
    # Sort by score descending
    top_k_idx = np.argsort(y_score)[::-1][:k]
    top_k_true = y_true[top_k_idx]
    return float(np.mean(top_k_true))


def coverage_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int = 5
) -> float:
    """Compute Coverage@K (recall of positives in top-K).

    Args:
        y_true: Ground truth labels
        y_score: Prediction scores
        k: Top-K

    Returns:
        Coverage@K (fraction of positives in top-K)
    """
    if np.sum(y_true) == 0:
        return 0.0

    top_k_idx = np.argsort(y_score)[::-1][:k]
    top_k_true = y_true[top_k_idx]
    return float(np.sum(top_k_true) / np.sum(y_true))


def recall_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int = 20
) -> float:
    """Compute Recall@K (for retrieval evaluation).

    Args:
        y_true: Ground truth labels
        y_score: Prediction scores
        k: Top-K

    Returns:
        Recall@K
    """
    return coverage_at_k(y_true, y_score, k)


def evaluate_evidence_predictions(
    predictions: List[Dict],
    k_values: List[int] = [5, 10, 20],
    prob_key: str = "prob",
) -> Dict:
    """Evaluate evidence predictions.

    Args:
        predictions: List of predictions with keys:
            - post_id, sent_id, cid, label, logit, prob (or prob_key)
        k_values: List of K values for Precision@K and Coverage@K
        prob_key: Key to use for probability values

    Returns:
        Dict with metrics
    """
    # Extract arrays
    y_true = np.array([p["label"] for p in predictions])
    y_prob = np.array([p.get(prob_key, p.get("prob", 0.0)) for p in predictions])

    # Binary predictions (threshold 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    # Overall metrics
    metrics = {
        "n_predictions": len(predictions),
        "n_positives": int(np.sum(y_true)),
        "n_negatives": int(len(y_true) - np.sum(y_true)),
    }

    # Classification metrics
    metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = 0.0  # Handle case with only one class
        
    metrics["f1"] = float(f1_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro"))

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    metrics["auc_pr"] = float(auc(recall, precision))

    # Precision@K, Coverage@K
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(y_true, y_prob, k)
        metrics[f"coverage@{k}"] = coverage_at_k(y_true, y_prob, k)
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_prob, k)

    # Calibration (ECE)
    metrics["ece"] = compute_ece(y_prob.tolist(), y_true.tolist(), bins=15)

    return metrics


def evaluate_per_criterion(
    predictions: List[Dict],
    k_values: List[int] = [5, 10, 20],
    prob_key: str = "prob",
) -> Dict[str, Dict]:
    """Evaluate evidence predictions per criterion.

    Args:
        predictions: List of predictions
        k_values: List of K values
        prob_key: Key to use for probability values

    Returns:
        Dict mapping criterion ID to metrics
    """
    # Group by criterion
    by_crit: Dict[str, List[Dict]] = defaultdict(list)
    for pred in predictions:
        by_crit[pred["cid"]].append(pred)

    # Evaluate each criterion
    per_crit_metrics = {}
    for cid, crit_preds in by_crit.items():
        per_crit_metrics[cid] = evaluate_evidence_predictions(crit_preds, k_values, prob_key)

    return per_crit_metrics


def compare_calibration(
    predictions_before: List[Dict],
    predictions_after: List[Dict],
) -> Dict:
    """Compare calibration before/after temperature scaling.

    Args:
        predictions_before: Predictions before temperature scaling
        predictions_after: Predictions after temperature scaling

    Returns:
        Dict with before/after ECE and improvement
    """
    y_true = np.array([p["label"] for p in predictions_before])
    y_prob_before = np.array([p["prob"] for p in predictions_before])
    y_prob_after = np.array([p["prob"] for p in predictions_after])

    ece_before = compute_ece(y_prob_before.tolist(), y_true.tolist())
    ece_after = compute_ece(y_prob_after.tolist(), y_true.tolist())

    improvement = ece_before - ece_after
    improvement_pct = (improvement / ece_before * 100) if ece_before > 0 else 0

    return {
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "improvement": float(improvement),
        "improvement_pct": float(improvement_pct),
    }


def save_evidence_metrics(
    predictions: List[Dict],
    output_path: Path,
    k_values: List[int] = [5, 10, 20],
) -> None:
    """Evaluate and save evidence metrics to JSON.

    Args:
        predictions: List of predictions
        output_path: Path to save metrics JSON
        k_values: List of K values
    """
    # Overall metrics (calibrated by default)
    overall_metrics = evaluate_evidence_predictions(predictions, k_values, prob_key="prob_cal")
    
    # Per-criterion metrics
    per_crit_metrics = evaluate_per_criterion(predictions, k_values, prob_key="prob_cal")

    # Combine
    metrics = {
        "overall": overall_metrics,
        "per_criterion": per_crit_metrics,
    }
    
    # If prob_uncal is present, evaluate that too for comparison
    if predictions and "prob_uncal" in predictions[0]:
        overall_uncal = evaluate_evidence_predictions(predictions, k_values, prob_key="prob_uncal")
        metrics["overall_uncalibrated"] = overall_uncal
        
        # Calculate explicit ECE improvement
        metrics["calibration_improvement"] = {
            "ece_before": overall_uncal["ece"],
            "ece_after": overall_metrics["ece"],
            "improvement": overall_uncal["ece"] - overall_metrics["ece"],
            "improvement_pct": (overall_uncal["ece"] - overall_metrics["ece"]) / overall_uncal["ece"] * 100 if overall_uncal["ece"] > 0 else 0
        }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evidence metrics to {output_path}")
    print(f"Overall AUPRC: {overall_metrics['auprc']:.4f}")
    print(f"Overall F1: {overall_metrics['f1']:.4f}")
    print(f"Overall ECE: {overall_metrics['ece']:.4f}")
    if "calibration_improvement" in metrics:
        print(f"ECE Improvement: {metrics['calibration_improvement']['improvement_pct']:.1f}%")
