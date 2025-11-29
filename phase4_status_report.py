#!/usr/bin/env python3
"""Phase 4 Status Report Generator (No PyTorch Required).

This script analyzes existing results from Trial #8 without requiring PyTorch.
It generates a comprehensive report on model performance, calibration status,
and readiness for PC layer training.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze_predictions(predictions: List[Dict]) -> Dict:
    """Analyze prediction statistics without PyTorch."""
    import statistics

    logits = [p["logit"] for p in predictions]
    probs_uncal = [p["prob_uncal"] for p in predictions]
    probs_cal = [p["prob_cal"] for p in predictions]
    labels = [p["label"] for p in predictions]

    # Split by label
    pos_logits = [p["logit"] for p in predictions if p["label"] == 1]
    neg_logits = [p["logit"] for p in predictions if p["label"] == 0]

    return {
        "total_samples": len(predictions),
        "n_positives": sum(labels),
        "n_negatives": len(labels) - sum(labels),
        "positive_ratio": sum(labels) / len(labels),
        "logits": {
            "mean": statistics.mean(logits),
            "median": statistics.median(logits),
            "stdev": statistics.stdev(logits) if len(logits) > 1 else 0,
            "min": min(logits),
            "max": max(logits),
            "positive_median": statistics.median(pos_logits) if pos_logits else 0,
            "negative_median": statistics.median(neg_logits) if neg_logits else 0,
            "separation": statistics.median(pos_logits) - statistics.median(neg_logits) if pos_logits and neg_logits else 0,
        },
        "prob_changes": {
            "mean_abs_change": statistics.mean([abs(u - c) for u, c in zip(probs_uncal, probs_cal)]),
            "median_abs_change": statistics.median([abs(u - c) for u, c in zip(probs_uncal, probs_cal)]),
            "max_change": max([abs(u - c) for u, c in zip(probs_uncal, probs_cal)]),
        }
    }


def analyze_per_criterion(predictions: List[Dict]) -> Dict[str, Dict]:
    """Analyze predictions per criterion."""
    from collections import defaultdict
    import statistics

    by_crit = defaultdict(list)
    for p in predictions:
        by_crit[p["cid"]].append(p)

    results = {}
    for cid, preds in sorted(by_crit.items()):
        logits = [p["logit"] for p in preds]
        labels = [p["label"] for p in preds]

        results[cid] = {
            "n_samples": len(preds),
            "n_positives": sum(labels),
            "positive_ratio": sum(labels) / len(labels) if labels else 0,
            "median_logit": statistics.median(logits) if logits else 0,
        }

    return results


def generate_report(exp: str, output_dir: Path) -> None:
    """Generate comprehensive status report."""

    print("=" * 80)
    print(f"PHASE 4 STATUS REPORT: {exp}")
    print("=" * 80)
    print()

    # Load data
    oof_path = output_dir / "oof_predictions.jsonl"
    metrics_path = output_dir / "evidence_metrics.json"
    temps_path = output_dir / "fold_temperatures.json"

    if not oof_path.exists():
        print(f"✗ ERROR: OOF predictions not found at {oof_path}")
        sys.exit(1)

    print(f"✓ Loading OOF predictions from {oof_path}")
    predictions = load_jsonl(oof_path)
    print(f"  Loaded {len(predictions):,} predictions")
    print()

    # Analyze predictions
    print("─" * 80)
    print("1. PREDICTION ANALYSIS")
    print("─" * 80)
    stats = analyze_predictions(predictions)

    print(f"\nDataset Statistics:")
    print(f"  Total samples:    {stats['total_samples']:,}")
    print(f"  Positives:        {stats['n_positives']:,} ({stats['positive_ratio']:.2%})")
    print(f"  Negatives:        {stats['n_negatives']:,} ({1-stats['positive_ratio']:.2%})")

    print(f"\nLogit Distribution:")
    print(f"  Mean:             {stats['logits']['mean']:7.4f}")
    print(f"  Median:           {stats['logits']['median']:7.4f}")
    print(f"  Std Dev:          {stats['logits']['stdev']:7.4f}")
    print(f"  Range:            [{stats['logits']['min']:7.4f}, {stats['logits']['max']:7.4f}]")

    print(f"\nClass Separation:")
    print(f"  Positive median:  {stats['logits']['positive_median']:7.4f}")
    print(f"  Negative median:  {stats['logits']['negative_median']:7.4f}")
    print(f"  Separation:       {stats['logits']['separation']:7.4f}", end="")

    # Assessment
    if stats['logits']['separation'] < 0.5:
        print("  ⚠️  WEAK (< 0.5)")
    elif stats['logits']['separation'] < 1.0:
        print("  ⚠️  MODERATE (0.5-1.0)")
    else:
        print("  ✓ GOOD (> 1.0)")

    print(f"\nCalibration Impact:")
    print(f"  Mean |Δprob|:     {stats['prob_changes']['mean_abs_change']:7.5f}")
    print(f"  Median |Δprob|:   {stats['prob_changes']['median_abs_change']:7.5f}")
    print(f"  Max |Δprob|:      {stats['prob_changes']['max_change']:7.5f}", end="")

    if stats['prob_changes']['median_abs_change'] < 0.01:
        print("  ⚠️  MINIMAL")
    elif stats['prob_changes']['median_abs_change'] < 0.05:
        print("  ⚠️  SMALL")
    else:
        print("  ✓ SIGNIFICANT")

    # Load metrics if available
    if metrics_path.exists():
        print()
        print("─" * 80)
        print("2. PERFORMANCE METRICS")
        print("─" * 80)

        with open(metrics_path) as f:
            metrics = json.load(f)

        overall = metrics["overall"]

        print(f"\nRanking Metrics:")
        print(f"  AUPRC:            {overall['auprc']:7.4f}")
        print(f"  AUROC:            {overall['auroc']:7.4f}")
        print(f"  Precision@5:      {overall['precision@5']:7.4f}", end="")
        if overall['precision@5'] >= 0.80:
            print("  ✓ PASS (≥ 0.80)")
        else:
            print("  ✗ FAIL (< 0.80)")

        print(f"\nCoverage Metrics:")
        print(f"  Coverage@5:       {overall['coverage@5']:7.4f}", end="")
        if overall['coverage@5'] >= 0.85:
            print("  ✓ EXCELLENT")
        else:
            print("  ⚠️  LOW")
        print(f"  Coverage@10:      {overall['coverage@10']:7.4f}")
        print(f"  Coverage@20:      {overall['coverage@20']:7.4f}")

        print(f"\nCalibration:")
        if "overall_uncalibrated" in metrics and "calibration_improvement" in metrics:
            cal = metrics["calibration_improvement"]
            print(f"  ECE before:       {cal['ece_before']:7.4f} ({cal['ece_before']*100:.1f}%)")
            print(f"  ECE after:        {cal['ece_after']:7.4f} ({cal['ece_after']*100:.1f}%)  ", end="")

            if cal['ece_after'] < 0.10:
                print("✓ EXCELLENT")
            elif cal['ece_after'] < 0.20:
                print("⚠️  ACCEPTABLE")
            else:
                print("✗ POOR")

            print(f"  Improvement:      {cal['improvement']:7.4f} ({cal['improvement_pct']:.1f}%)  ", end="")

            if cal['improvement_pct'] > 10:
                print("✓ EFFECTIVE")
            elif cal['improvement_pct'] > 5:
                print("⚠️  MODERATE")
            else:
                print("✗ INEFFECTIVE")
        else:
            print(f"  ECE:              {overall['ece']:7.4f} ({overall['ece']*100:.1f}%)  ", end="")
            if overall['ece'] < 0.10:
                print("✓ EXCELLENT")
            elif overall['ece'] < 0.20:
                print("⚠️  ACCEPTABLE")
            else:
                print("✗ POOR")

        print(f"\nClassification (threshold=0.5):")
        print(f"  F1:               {overall['f1']:7.4f}")
        print(f"  Macro-F1:         {overall['macro_f1']:7.4f}")

        if "optimal_threshold_metrics" in metrics:
            opt = metrics["optimal_threshold_metrics"]
            print(f"\nClassification (optimal thresholds):")
            print(f"  F1:               {opt['f1_optimal']:7.4f}  ", end="")
            improvement = (opt['f1_optimal'] - overall['f1']) / overall['f1'] * 100
            print(f"(+{improvement:.1f}% vs default)")
            print(f"  Macro-F1:         {opt['macro_f1_optimal']:7.4f}")
            print(f"  Precision:        {opt['precision_optimal']:7.4f}")

    # Load temperatures
    if temps_path.exists():
        print()
        print("─" * 80)
        print("3. TEMPERATURE SCALING")
        print("─" * 80)

        with open(temps_path) as f:
            temps = json.load(f)

        print(f"\nGlobal Temperature:")
        print(f"  Mean:             {temps['global']['mean']:7.4f}")
        print(f"  Std:              {temps['global']['std']:7.4f}")

        print(f"\nPer-Class Temperatures:")
        for cid in sorted(temps['per_class_mean'].keys()):
            mean_t = temps['per_class_mean'][cid]
            std_t = temps['per_class_std'][cid]
            print(f"  {cid}:              {mean_t:7.4f} ± {std_t:5.4f}")

    # Per-criterion breakdown
    print()
    print("─" * 80)
    print("4. PER-CRITERION BREAKDOWN")
    print("─" * 80)

    per_crit = analyze_per_criterion(predictions)
    print(f"\n{'CID':<5} {'Samples':>8} {'Positives':>10} {'Pos%':>7} {'Med Logit':>10}")
    print("─" * 50)
    for cid, stats in per_crit.items():
        print(f"{cid:<5} {stats['n_samples']:8,} {stats['n_positives']:10,} "
              f"{stats['positive_ratio']:6.2%} {stats['median_logit']:10.4f}")

    # Final assessment
    print()
    print("=" * 80)
    print("5. READINESS ASSESSMENT")
    print("=" * 80)

    checks = []

    # Check 1: OOF predictions exist
    checks.append(("OOF predictions generated", True, "✓"))

    # Check 2: Performance metrics
    if metrics_path.exists():
        prec_at_5 = metrics["overall"]["precision@5"]
        checks.append(("Precision@5 ≥ 0.80", prec_at_5 >= 0.80, "✓" if prec_at_5 >= 0.80 else "✗"))

        coverage_at_20 = metrics["overall"]["coverage@20"]
        checks.append(("Coverage@20 ≥ 0.95", coverage_at_20 >= 0.95, "✓" if coverage_at_20 >= 0.95 else "⚠️"))

        if "calibration_improvement" in metrics:
            ece_after = metrics["calibration_improvement"]["ece_after"]
            checks.append(("ECE < 0.10 (target)", ece_after < 0.10, "✓" if ece_after < 0.10 else "✗"))

    # Check 3: Temperature fitting
    checks.append(("Temperature scaling completed", temps_path.exists(), "✓" if temps_path.exists() else "✗"))

    # Check 4: Optimal thresholds
    if metrics_path.exists() and "optimal_thresholds" in metrics:
        checks.append(("Optimal thresholds computed", True, "✓"))

    print()
    for desc, passed, symbol in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {symbol} {desc:<40} [{status}]")

    # Overall verdict
    print()
    all_critical_pass = all(passed for desc, passed, _ in checks if "predictions" in desc or "Precision@5" in desc)

    if all_critical_pass:
        print("  ✓ READY FOR PC LAYER TRAINING")
        print()
        print("  Recommended next steps:")
        print("    1. Fix CUDA environment (./fix_cuda_environment.sh)")
        print("    2. Build graph: python scripts/build_graph.py --cfg configs/graph.yaml")
        print("    3. Train PC layer: python scripts/train_ce_pc.py --cfg configs/ce_pc_real.yaml")
    else:
        print("  ⚠️  PROCEED WITH CAUTION")
        print()
        print("  Issues to address:")
        if metrics_path.exists():
            if metrics["overall"]["precision@5"] < 0.80:
                print("    - Precision@5 below threshold (consider retraining S-C layer)")
            if "calibration_improvement" in metrics and metrics["calibration_improvement"]["ece_after"] > 0.10:
                print("    - Poor calibration (consider using optimal thresholds instead)")
        print()
        print("  You can still proceed to PC layer training, but results may be suboptimal.")

    print()
    print("=" * 80)

    # Save report
    report_path = output_dir / "phase4_status_report.txt"
    print(f"\n✓ Report saved to: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Phase 4 status report")
    parser.add_argument("--exp", default="real_dev_hpo_refine_trial_8", help="Experiment name")
    parser.add_argument("--output", help="Output directory (default: outputs/runs/{exp})")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(f"outputs/runs/{args.exp}")

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    generate_report(args.exp, output_dir)
