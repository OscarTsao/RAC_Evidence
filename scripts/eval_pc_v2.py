"""Evaluate PC v2 OOF predictions with per-criterion and macro-averaged metrics.

Computes:
- Per-criterion AUPRC, F1 (at optimal threshold), Precision, Recall
- Macro-averaged metrics
- Saves results to metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from Project.utils.logging import get_logger


def load_predictions(oof_path: Path) -> List[Dict]:
    """Load OOF predictions from JSONL file.

    Args:
        oof_path: Path to pc_oof.jsonl

    Returns:
        List of prediction dicts
    """
    predictions = []
    with open(oof_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def compute_optimal_threshold_f1(labels: List[int], probs: List[float]) -> tuple:
    """Find threshold that maximizes F1 score.

    Args:
        labels: Ground truth labels
        probs: Predicted probabilities

    Returns:
        Tuple of (optimal_threshold, best_f1)
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    # Compute F1 for each threshold
    f1_scores = []
    for p, r in zip(precision, recall):
        if p + r == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * p * r / (p + r))

    # Find best F1
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]

    # Get threshold (note: thresholds has one fewer element)
    if best_idx < len(thresholds):
        optimal_threshold = thresholds[best_idx]
    else:
        optimal_threshold = 1.0

    return optimal_threshold, best_f1


def evaluate_per_criterion(predictions: List[Dict]) -> Dict[str, Dict]:
    """Compute per-criterion metrics.

    Args:
        predictions: List of prediction dicts

    Returns:
        Dict mapping criterion ID to metrics dict
    """
    logger = get_logger(__name__)

    # Group predictions by criterion
    per_cid = defaultdict(list)
    for pred in predictions:
        cid = pred["cid"]
        per_cid[cid].append(pred)

    results = {}

    for cid in sorted(per_cid.keys()):
        preds = per_cid[cid]
        labels = [p["label"] for p in preds]
        probs_cal = [p["prob_cal"] for p in preds]
        probs_uncal = [p["prob_uncal"] for p in preds]

        # Check if we have both classes
        if len(set(labels)) < 2:
            logger.warning(f"Criterion {cid} has only one class (all {labels[0]}), skipping AUPRC")
            auprc_cal = auprc_uncal = None
        else:
            auprc_cal = average_precision_score(labels, probs_cal)
            auprc_uncal = average_precision_score(labels, probs_uncal)

        # Find optimal threshold
        optimal_thresh, optimal_f1 = compute_optimal_threshold_f1(labels, probs_cal)

        # Compute metrics at optimal threshold
        preds_optimal = [1 if p >= optimal_thresh else 0 for p in probs_cal]
        precision_optimal = precision_score(labels, preds_optimal, zero_division=0)
        recall_optimal = recall_score(labels, preds_optimal, zero_division=0)

        # Compute metrics at fixed threshold 0.5
        preds_05 = [1 if p >= 0.5 else 0 for p in probs_cal]
        f1_05 = f1_score(labels, preds_05, zero_division=0)
        precision_05 = precision_score(labels, preds_05, zero_division=0)
        recall_05 = recall_score(labels, preds_05, zero_division=0)

        results[cid] = {
            "n_samples": len(preds),
            "n_pos": sum(labels),
            "n_neg": len(labels) - sum(labels),
            "auprc_cal": float(auprc_cal) if auprc_cal is not None else None,
            "auprc_uncal": float(auprc_uncal) if auprc_uncal is not None else None,
            "optimal_threshold": float(optimal_thresh),
            "f1_optimal": float(optimal_f1),
            "precision_optimal": float(precision_optimal),
            "recall_optimal": float(recall_optimal),
            "f1_at_0.5": float(f1_05),
            "precision_at_0.5": float(precision_05),
            "recall_at_0.5": float(recall_05),
        }

        logger.info(
            f"Criterion {cid}: AUPRC={auprc_cal:.4f if auprc_cal else 'N/A'}, "
            f"F1_optimal={optimal_f1:.4f} (thresh={optimal_thresh:.3f}), "
            f"F1@0.5={f1_05:.4f}"
        )

    return results


def compute_macro_metrics(per_cid_results: Dict[str, Dict]) -> Dict:
    """Compute macro-averaged metrics across criteria.

    Args:
        per_cid_results: Per-criterion metrics

    Returns:
        Dict of macro-averaged metrics
    """
    # Extract valid AUPRC values (not None)
    auprc_cal_values = [
        r["auprc_cal"] for r in per_cid_results.values() if r["auprc_cal"] is not None
    ]
    auprc_uncal_values = [
        r["auprc_uncal"] for r in per_cid_results.values() if r["auprc_uncal"] is not None
    ]

    # Extract optimal threshold metrics
    f1_optimal_values = [r["f1_optimal"] for r in per_cid_results.values()]
    precision_optimal_values = [r["precision_optimal"] for r in per_cid_results.values()]
    recall_optimal_values = [r["recall_optimal"] for r in per_cid_results.values()]

    # Extract fixed threshold metrics
    f1_05_values = [r["f1_at_0.5"] for r in per_cid_results.values()]
    precision_05_values = [r["precision_at_0.5"] for r in per_cid_results.values()]
    recall_05_values = [r["recall_at_0.5"] for r in per_cid_results.values()]

    return {
        "macro_auprc_cal": float(np.mean(auprc_cal_values)) if auprc_cal_values else None,
        "macro_auprc_uncal": (
            float(np.mean(auprc_uncal_values)) if auprc_uncal_values else None
        ),
        "macro_f1_optimal": float(np.mean(f1_optimal_values)),
        "macro_precision_optimal": float(np.mean(precision_optimal_values)),
        "macro_recall_optimal": float(np.mean(recall_optimal_values)),
        "macro_f1_at_0.5": float(np.mean(f1_05_values)),
        "macro_precision_at_0.5": float(np.mean(precision_05_values)),
        "macro_recall_at_0.5": float(np.mean(recall_05_values)),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate PC v2 OOF predictions")
    parser.add_argument(
        "--oof_path",
        type=str,
        default="outputs/runs/real_dev_pc_v2/pc_oof.jsonl",
        help="Path to OOF predictions",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/runs/real_dev_pc_v2/metrics.json",
        help="Path to save metrics",
    )
    args = parser.parse_args()

    logger = get_logger(__name__)

    # Load predictions
    oof_path = Path(args.oof_path)
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")

    predictions = load_predictions(oof_path)
    logger.info(f"Loaded {len(predictions)} OOF predictions from {oof_path}")

    # Compute per-criterion metrics
    per_cid_results = evaluate_per_criterion(predictions)

    # Compute macro-averaged metrics
    macro_results = compute_macro_metrics(per_cid_results)

    # Log macro metrics
    logger.info("=" * 60)
    logger.info("Macro-averaged metrics:")
    logger.info(f"  AUPRC (calibrated): {macro_results['macro_auprc_cal']:.4f}")
    logger.info(f"  F1 (optimal thresh): {macro_results['macro_f1_optimal']:.4f}")
    logger.info(f"  Precision (optimal): {macro_results['macro_precision_optimal']:.4f}")
    logger.info(f"  Recall (optimal): {macro_results['macro_recall_optimal']:.4f}")
    logger.info(f"  F1 (fixed @ 0.5): {macro_results['macro_f1_at_0.5']:.4f}")
    logger.info("=" * 60)

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "per_criterion": per_cid_results,
        "macro": macro_results,
        "n_predictions": len(predictions),
        "n_criteria": len(per_cid_results),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
