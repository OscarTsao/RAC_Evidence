"""Evaluate PC v2 OOF predictions against ground truth."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def load_jsonl(path: Path):
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_optimal_threshold(y_true, y_prob):
    """Find optimal threshold maximizing F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold, f1_scores[best_idx]


def evaluate_pc_v2(
    predictions_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    threshold: float = 0.5,
):
    """Evaluate PC v2 OOF predictions."""

    # Load data
    print(f"Loading predictions from {predictions_path}...")
    predictions = load_jsonl(predictions_path)

    print(f"Loading ground truth from {ground_truth_path}...")
    ground_truth = load_jsonl(ground_truth_path)

    # Create ground truth lookup
    gt_lookup = {(item["post_id"], item["cid"]): item["label"] for item in ground_truth}

    # Align predictions with ground truth
    aligned_preds = []
    for pred in predictions:
        key = (pred["post_id"], pred["cid"])
        if key in gt_lookup:
            aligned_preds.append({
                **pred,
                "gt_label": gt_lookup[key]
            })

    print(f"\nAligned {len(aligned_preds)} predictions with ground truth")

    # Extract labels and probabilities
    y_true = np.array([p["gt_label"] for p in aligned_preds])
    y_prob_uncal = np.array([p["prob_uncal"] for p in aligned_preds])
    y_prob_cal = np.array([p["prob_cal"] for p in aligned_preds])
    logits = np.array([p["logit"] for p in aligned_preds])
    cids = [p["cid"] for p in aligned_preds]
    folds = [p["fold"] for p in aligned_preds]

    # Predictions at fixed threshold
    y_pred_uncal = (y_prob_uncal >= threshold).astype(int)
    y_pred_cal = (y_prob_cal >= threshold).astype(int)

    # Overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS (threshold=0.5)")
    print("="*80)

    print("\nUncalibrated Probabilities:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred_uncal):.4f}")
    print(f"  Precision: {classification_report(y_true, y_pred_uncal, output_dict=True)['1']['precision']:.4f}")
    print(f"  Recall:    {classification_report(y_true, y_pred_uncal, output_dict=True)['1']['recall']:.4f}")
    print(f"  F1:        {f1_score(y_true, y_pred_uncal):.4f}")
    try:
        print(f"  ROC-AUC:   {roc_auc_score(y_true, y_prob_uncal):.4f}")
    except ValueError:
        print("  ROC-AUC:   N/A (only one class present)")

    print("\nCalibrated Probabilities:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred_cal):.4f}")
    print(f"  Precision: {classification_report(y_true, y_pred_cal, output_dict=True)['1']['precision']:.4f}")
    print(f"  Recall:    {classification_report(y_true, y_pred_cal, output_dict=True)['1']['recall']:.4f}")
    print(f"  F1:        {f1_score(y_true, y_pred_cal):.4f}")
    try:
        print(f"  ROC-AUC:   {roc_auc_score(y_true, y_prob_cal):.4f}")
    except ValueError:
        print("  ROC-AUC:   N/A (only one class present)")

    # Per-class metrics
    print("\n" + "="*80)
    print("PER-CRITERION METRICS (Calibrated)")
    print("="*80)

    per_class_metrics = {}
    for cid in sorted(set(cids)):
        mask = np.array([c == cid for c in cids])
        y_true_c = y_true[mask]
        y_prob_c = y_prob_cal[mask]
        y_pred_c = (y_prob_c >= threshold).astype(int)

        n_pos = y_true_c.sum()
        n_neg = len(y_true_c) - n_pos

        # Skip if no positive samples
        if n_pos == 0:
            print(f"\n{cid}: SKIPPED (no positive samples)")
            continue

        acc = accuracy_score(y_true_c, y_pred_c)
        f1 = f1_score(y_true_c, y_pred_c)

        report = classification_report(y_true_c, y_pred_c, output_dict=True)
        prec = report['1']['precision']
        rec = report['1']['recall']

        try:
            auc = roc_auc_score(y_true_c, y_prob_c)
        except ValueError:
            auc = None

        # Find optimal threshold
        opt_thresh, opt_f1 = compute_optimal_threshold(y_true_c, y_prob_c)

        per_class_metrics[cid] = {
            "n_samples": len(y_true_c),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "pos_rate": float(n_pos / len(y_true_c)),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc) if auc is not None else None,
            "optimal_threshold": float(opt_thresh),
            "optimal_f1": float(opt_f1),
        }

        print(f"\n{cid}:")
        print(f"  Samples:   {len(y_true_c):5d} (pos={n_pos:4d}, neg={n_neg:5d}, {n_pos/len(y_true_c)*100:.1f}%)")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        if auc is not None:
            print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Optimal:   thresh={opt_thresh:.4f}, F1={opt_f1:.4f}")

    # Compute macro-averaged F1
    macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])
    macro_f1_optimal = np.mean([m["optimal_f1"] for m in per_class_metrics.values()])

    print("\n" + "="*80)
    print("MACRO-AVERAGED METRICS")
    print("="*80)
    print(f"  Macro F1 (threshold=0.5):  {macro_f1:.4f}")
    print(f"  Macro F1 (optimal thresh): {macro_f1_optimal:.4f}")

    # Per-fold metrics
    print("\n" + "="*80)
    print("PER-FOLD METRICS (Calibrated)")
    print("="*80)

    per_fold_metrics = {}
    for fold_id in sorted(set(folds)):
        mask = np.array([f == fold_id for f in folds])
        y_true_f = y_true[mask]
        y_prob_f = y_prob_cal[mask]
        y_pred_f = (y_prob_f >= threshold).astype(int)

        acc = accuracy_score(y_true_f, y_pred_f)
        f1 = f1_score(y_true_f, y_pred_f)

        per_fold_metrics[fold_id] = {
            "n_samples": len(y_true_f),
            "n_pos": int(y_true_f.sum()),
            "accuracy": float(acc),
            "f1": float(f1),
        }

        print(f"\nFold {fold_id}:")
        print(f"  Samples:  {len(y_true_f):5d} (pos={y_true_f.sum():4d})")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1:       {f1:.4f}")

    # Confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX (Calibrated, threshold=0.5)")
    print("="*80)
    cm = confusion_matrix(y_true, y_pred_cal)
    print(f"\n           Predicted")
    print(f"           Neg    Pos")
    print(f"Actual Neg {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Pos {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Calibration analysis
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS")
    print("="*80)

    # Read temperature parameters
    temps_path = predictions_path.parent / "fold_temperatures.json"
    if temps_path.exists():
        with open(temps_path) as f:
            temps = json.load(f)

        print(f"\nGlobal Temperature:")
        print(f"  Mean: {temps['global']['mean']:.4f}")
        print(f"  Std:  {temps['global']['std']:.4f}")
        print(f"  Per-fold: {[f'{t:.4f}' for t in temps['global']['per_fold']]}")

        print(f"\nPer-Class Temperatures:")
        for cid in sorted(temps.get('per_class_mean', {}).keys()):
            mean = temps['per_class_mean'][cid]
            std = temps['per_class_std'][cid]
            print(f"  {cid}: mean={mean:.4f}, std={std:.4f}")

    # Save results
    results = {
        "overall": {
            "uncalibrated": {
                "accuracy": float(accuracy_score(y_true, y_pred_uncal)),
                "f1": float(f1_score(y_true, y_pred_uncal)),
                "roc_auc": float(roc_auc_score(y_true, y_prob_uncal)) if len(set(y_true)) > 1 else None,
            },
            "calibrated": {
                "accuracy": float(accuracy_score(y_true, y_pred_cal)),
                "f1": float(f1_score(y_true, y_pred_cal)),
                "roc_auc": float(roc_auc_score(y_true, y_prob_cal)) if len(set(y_true)) > 1 else None,
            },
            "macro_f1": float(macro_f1),
            "macro_f1_optimal": float(macro_f1_optimal),
        },
        "per_criterion": per_class_metrics,
        "per_fold": per_fold_metrics,
        "confusion_matrix": cm.tolist(),
        "class_distribution": {
            "n_samples": len(y_true),
            "n_pos": int(y_true.sum()),
            "n_neg": int(len(y_true) - y_true.sum()),
            "pos_rate": float(y_true.mean()),
        },
    }

    print(f"\nSaving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation complete!")
    return results


if __name__ == "__main__":
    predictions_path = Path("outputs/runs/real_dev_pc_v2/pc_oof.jsonl")
    ground_truth_path = Path("data/processed/labels_pc.jsonl")
    output_path = Path("outputs/runs/real_dev_pc_v2/evaluation_results.json")

    evaluate_pc_v2(predictions_path, ground_truth_path, output_path)
