"""Evaluate PC predictions against ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

from Project.utils.logging import get_logger


# Mapping from DSM5 symptoms to criterion IDs
SYMPTOM_TO_CID = {
    "A.1": "c1",  # Depressed Mood
    "A.2": "c2",  # Anhedonia
    "A.3": "c3",  # Appetite Change
    "A.4": "c4",  # Sleep Issues
    "A.5": "c5",  # Psychomotor
    "A.6": "c6",  # Fatigue
    "A.7": "c7",  # Worthlessness
    "A.8": "c8",  # Cognitive Issues
    "A.9": "c9",  # Suicidal Thoughts
    # A.10 excluded - special case not in our criteria
}


def load_groundtruth(gt_path: Path) -> pd.DataFrame:
    """Load and process ground truth labels."""
    logger = get_logger(__name__)
    logger.info(f"Loading ground truth from {gt_path}")

    df = pd.read_csv(gt_path)
    logger.info(f"Loaded {len(df)} ground truth samples")

    # Convert DSM5_symptom to cid
    df["cid"] = df["DSM5_symptom"].map(SYMPTOM_TO_CID)

    # Filter out A.10 (special case)
    df = df[df["cid"].notna()].copy()
    logger.info(f"Filtered to {len(df)} samples (excluded A.10)")

    # Keep only needed columns
    df = df[["post_id", "cid", "groundtruth"]].copy()

    return df


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """Load PC predictions."""
    logger = get_logger(__name__)
    logger.info(f"Loading predictions from {pred_path}")

    predictions = []
    with open(pred_path) as f:
        for line in f:
            predictions.append(json.loads(line))

    df = pd.DataFrame(predictions)
    logger.info(f"Loaded {len(df)} predictions")

    return df


def evaluate_pc(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """Evaluate PC predictions against ground truth."""
    logger = get_logger(__name__)

    # Merge predictions with ground truth
    merged = pd.merge(
        pred_df,
        gt_df,
        on=["post_id", "cid"],
        how="inner",
    )

    logger.info(f"Matched {len(merged)} samples")

    y_true = merged["groundtruth"].values
    y_pred_prob = merged["prob"].values
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Overall metrics
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)

    # Per-criterion metrics
    per_cid_metrics = {}
    for cid in sorted(merged["cid"].unique()):
        mask = merged["cid"] == cid
        cid_y_true = y_true[mask]
        cid_y_pred = y_pred[mask]
        cid_y_prob = y_pred_prob[mask]

        # Skip if only one class present
        if len(np.unique(cid_y_true)) == 1:
            per_cid_metrics[cid] = {
                "f1": 0.0,
                "auc": 0.0,
                "auprc": 0.0,
                "support": int(mask.sum()),
            }
            continue

        cid_f1 = f1_score(cid_y_true, cid_y_pred)
        cid_auc = roc_auc_score(cid_y_true, cid_y_prob)
        cid_auprc = average_precision_score(cid_y_true, cid_y_prob)

        per_cid_metrics[cid] = {
            "f1": float(cid_f1),
            "auc": float(cid_auc),
            "auprc": float(cid_auprc),
            "support": int(mask.sum()),
        }

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "positive"],
        output_dict=True,
    )

    results = {
        "overall": {
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
            "auc": float(auc),
            "auprc": float(auprc),
            "num_samples": len(merged),
        },
        "per_criterion": per_cid_metrics,
        "classification_report": report,
    }

    return results


def main():
    logger = get_logger(__name__)

    # Paths
    gt_path = Path("data/groundtruth/criteria_matching_groundtruth.csv")
    pred_path = Path("outputs/runs/real_dev/pc/oof_predictions.jsonl")
    output_dir = Path("outputs/runs/real_dev/pc")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    gt_df = load_groundtruth(gt_path)
    pred_df = load_predictions(pred_path)

    # Evaluate
    results = evaluate_pc(gt_df, pred_df)

    # Print results
    logger.info("\n=== PC Evaluation Results ===")
    logger.info(f"Overall F1 (micro): {results['overall']['f1_micro']:.4f}")
    logger.info(f"Overall F1 (macro): {results['overall']['f1_macro']:.4f}")
    logger.info(f"Overall AUC: {results['overall']['auc']:.4f}")
    logger.info(f"Overall AUPRC: {results['overall']['auprc']:.4f}")

    logger.info("\nPer-Criterion Metrics:")
    for cid, metrics in sorted(results["per_criterion"].items()):
        logger.info(
            f"  {cid}: F1={metrics['f1']:.4f}, "
            f"AUC={metrics['auc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, "
            f"support={metrics['support']}"
        )

    # Save results
    output_path = output_dir / "groundtruth_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
