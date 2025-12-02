"""Fit optimal thresholds on PC v2 OOF predictions."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from Project.calib.threshold_optimizer import ThresholdOptimizer
from Project.utils.io import read_jsonl
from Project.utils.logging import get_logger


def fit_pc_thresholds(
    oof_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    min_samples: int = 30,
):
    """Fit optimal per-class thresholds on PC OOF predictions.

    Args:
        oof_path: Path to OOF predictions JSONL
        ground_truth_path: Path to ground truth labels JSONL
        output_dir: Output directory for thresholds and metrics
        min_samples: Minimum samples per class for threshold optimization
    """
    logger = get_logger(__name__)

    # Load data
    logger.info(f"Loading OOF predictions from {oof_path}...")
    predictions = read_jsonl(oof_path)

    logger.info(f"Loading ground truth from {ground_truth_path}...")
    ground_truth = read_jsonl(ground_truth_path)

    # Create ground truth lookup
    gt_lookup = {(item["post_id"], item["cid"]): item["label"] for item in ground_truth}

    # Align predictions with ground truth
    aligned_preds = []
    for pred in predictions:
        key = (pred["post_id"], pred["cid"])
        if key in gt_lookup:
            aligned_preds.append({
                **pred,
                "label": gt_lookup[key]  # Use ground truth label
            })

    logger.info(f"Aligned {len(aligned_preds)} predictions with ground truth")

    # Fit optimal thresholds
    logger.info(f"Fitting optimal thresholds (min_samples={min_samples})...")
    optimizer = ThresholdOptimizer(metric="f1", min_samples=min_samples)
    thresholds = optimizer.fit(aligned_preds)

    logger.info("\nOptimal Thresholds:")
    for cid in sorted(thresholds.keys()):
        logger.info(f"  {cid}: {thresholds[cid]:.4f}")

    # Save thresholds
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = output_dir / "optimal_thresholds.json"
    optimizer.save(thresholds_path)
    logger.info(f"\nSaved thresholds to {thresholds_path}")

    # Evaluate with fixed threshold=0.5
    logger.info("\n" + "="*80)
    logger.info("EVALUATION WITH FIXED THRESHOLD=0.5")
    logger.info("="*80)

    y_true = np.array([p["label"] for p in aligned_preds])
    y_prob_cal = np.array([p["prob_cal"] for p in aligned_preds])
    y_pred_fixed = (y_prob_cal >= 0.5).astype(int)

    f1_fixed = f1_score(y_true, y_pred_fixed)

    # Per-class F1 with fixed threshold
    from collections import defaultdict
    by_class = defaultdict(list)
    for pred in aligned_preds:
        cid = pred["cid"]
        by_class[cid].append(pred)

    f1_fixed_per_class = {}
    for cid, class_preds in by_class.items():
        y_true_c = np.array([p["label"] for p in class_preds])
        y_prob_c = np.array([p["prob_cal"] for p in class_preds])
        y_pred_c = (y_prob_c >= 0.5).astype(int)
        f1_fixed_per_class[cid] = f1_score(y_true_c, y_pred_c)

    macro_f1_fixed = np.mean(list(f1_fixed_per_class.values()))

    logger.info(f"Overall F1: {f1_fixed:.4f}")
    logger.info(f"Macro F1:   {macro_f1_fixed:.4f}")

    # Evaluate with optimal thresholds
    logger.info("\n" + "="*80)
    logger.info("EVALUATION WITH OPTIMAL THRESHOLDS")
    logger.info("="*80)

    y_pred_optimal = optimizer.predict(aligned_preds, prob_key="prob_cal")
    f1_optimal = f1_score(y_true, y_pred_optimal)

    # Per-class F1 with optimal thresholds
    f1_optimal_per_class = {}
    for cid, class_preds in by_class.items():
        y_true_c = np.array([p["label"] for p in class_preds])
        y_pred_c = optimizer.predict(class_preds, prob_key="prob_cal")
        f1_optimal_per_class[cid] = f1_score(y_true_c, y_pred_c)

    macro_f1_optimal = np.mean(list(f1_optimal_per_class.values()))

    logger.info(f"Overall F1: {f1_optimal:.4f}")
    logger.info(f"Macro F1:   {macro_f1_optimal:.4f}")

    # Improvement summary
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT SUMMARY")
    logger.info("="*80)

    improvement_pct = (macro_f1_optimal - macro_f1_fixed) / macro_f1_fixed * 100
    logger.info(f"Macro F1 improvement: {macro_f1_fixed:.4f} → {macro_f1_optimal:.4f} (+{improvement_pct:.1f}%)")

    logger.info("\nPer-Class F1 Improvements:")
    for cid in sorted(f1_fixed_per_class.keys()):
        f1_before = f1_fixed_per_class[cid]
        f1_after = f1_optimal_per_class[cid]
        improvement = (f1_after - f1_before) / f1_before * 100 if f1_before > 0 else 0
        logger.info(f"  {cid}: {f1_before:.4f} → {f1_after:.4f} (+{improvement:.1f}%)")

    # Save metrics
    metrics = {
        "fixed_threshold": {
            "threshold": 0.5,
            "overall_f1": float(f1_fixed),
            "macro_f1": float(macro_f1_fixed),
            "per_class_f1": {k: float(v) for k, v in f1_fixed_per_class.items()},
        },
        "optimal_thresholds": {
            "thresholds": {k: float(v) for k, v in thresholds.items()},
            "overall_f1": float(f1_optimal),
            "macro_f1": float(macro_f1_optimal),
            "per_class_f1": {k: float(v) for k, v in f1_optimal_per_class.items()},
        },
        "improvement": {
            "macro_f1_improvement_pct": float(improvement_pct),
            "macro_f1_delta": float(macro_f1_optimal - macro_f1_fixed),
        },
    }

    metrics_path = output_dir / "threshold_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nSaved metrics to {metrics_path}")

    # Generate classification report with optimal thresholds
    logger.info("\n" + "="*80)
    logger.info("CLASSIFICATION REPORT (Optimal Thresholds)")
    logger.info("="*80)
    logger.info(classification_report(y_true, y_pred_optimal, target_names=["Negative", "Positive"]))

    return thresholds, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-path",
        type=Path,
        default=Path("outputs/runs/real_dev_pc_v2/pc_oof.jsonl"),
        help="Path to OOF predictions",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/processed/labels_pc.jsonl"),
        help="Path to ground truth labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/real_dev_pc_v2"),
        help="Output directory",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum samples per class",
    )

    args = parser.parse_args()

    fit_pc_thresholds(
        oof_path=args.oof_path,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        min_samples=args.min_samples,
    )
