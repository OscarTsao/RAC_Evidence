"""Quality gates for Evidence RAC pipeline acceptance.

Implements quality gate checks for:
- Retrieval quality (Recall@20)
- Evidence CE quality (AUPRC, F1, P@K, ECE)
- Temperature calibration improvement
- Data leakage detection
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from Project.utils.logging import get_logger


@dataclass
class QualityGate:
    """Quality gate specification."""

    name: str
    metric: str
    threshold: float
    comparison: str  # 'gte' (>=) or 'lte' (<=)
    description: str


@dataclass
class GateResult:
    """Quality gate check result."""

    gate: QualityGate
    value: float
    passed: bool
    message: str


class QualityGateError(Exception):
    """Raised when quality gates fail."""

    pass


# Define quality gates
RETRIEVAL_GATES = [
    QualityGate(
        name="retrieval_recall@20_overall",
        metric="recall@20",
        threshold=0.95,
        comparison="gte",
        description="Overall Recall@20 must be >= 0.95",
    ),
    QualityGate(
        name="retrieval_recall@20_per_criterion",
        metric="recall@20_min",
        threshold=0.90,
        comparison="gte",
        description="Per-criterion Recall@20 must be >= 0.90 for all criteria",
    ),
]

EVIDENCE_GATES = [
    QualityGate(
        name="evidence_auprc",
        metric="auprc",
        threshold=0.75,
        comparison="gte",
        description="AUPRC must be >= 0.75",
    ),
    QualityGate(
        name="evidence_f1",
        metric="f1",
        threshold=0.70,
        comparison="gte",
        description="F1 score must be >= 0.70",
    ),
    QualityGate(
        name="evidence_precision@5",
        metric="precision@5",
        threshold=0.85,
        comparison="gte",
        description="Precision@5 must be >= 0.85",
    ),
    QualityGate(
        name="evidence_coverage@5",
        metric="coverage@5",
        threshold=0.80,
        comparison="gte",
        description="Coverage@5 must be >= 0.80",
    ),
    QualityGate(
        name="evidence_ece",
        metric="ece",
        threshold=0.10,
        comparison="lte",
        description="ECE must be <= 0.10",
    ),
]

CALIBRATION_GATES = [
    QualityGate(
        name="calibration_improvement",
        metric="improvement_pct",
        threshold=20.0,
        comparison="gte",
        description="ECE improvement must be >= 20% if initial ECE > 0.05",
    ),
]


def check_gate(gate: QualityGate, value: float) -> GateResult:
    """Check if a value passes a quality gate.

    Args:
        gate: Quality gate specification
        value: Metric value to check

    Returns:
        GateResult with pass/fail status
    """
    if gate.comparison == "gte":
        passed = value >= gate.threshold
        symbol = ">="
    elif gate.comparison == "lte":
        passed = value <= gate.threshold
        symbol = "<="
    else:
        raise ValueError(f"Unknown comparison: {gate.comparison}")

    message = (
        f"{gate.name}: {value:.4f} {symbol} {gate.threshold:.4f} "
        f"({'PASS' if passed else 'FAIL'})"
    )

    return GateResult(gate=gate, value=value, passed=passed, message=message)


def check_retrieval_gates(metrics: Dict) -> List[GateResult]:
    """Check retrieval quality gates.

    Args:
        metrics: Retrieval metrics dict with overall and per_criterion keys

    Returns:
        List of GateResult objects
    """
    logger = get_logger(__name__)
    results = []

    overall = metrics.get("overall", {})
    per_criterion = metrics.get("per_criterion", {})

    # Check overall recall@20
    recall_20_overall = overall.get("recall@20", 0.0)
    results.append(check_gate(RETRIEVAL_GATES[0], recall_20_overall))

    # Check per-criterion recall@20 (min across all criteria)
    if per_criterion:
        recall_20_values = [
            crit.get("recall@20", 0.0) for crit in per_criterion.values()
        ]
        recall_20_min = min(recall_20_values) if recall_20_values else 0.0
        results.append(check_gate(RETRIEVAL_GATES[1], recall_20_min))
    else:
        logger.warning("No per-criterion metrics found for retrieval")

    return results


def check_evidence_gates(metrics: Dict) -> List[GateResult]:
    """Check evidence quality gates.

    Args:
        metrics: Evidence metrics dict with overall key

    Returns:
        List of GateResult objects
    """
    results = []

    overall = metrics.get("overall", {})

    for gate in EVIDENCE_GATES:
        value = overall.get(gate.metric, 0.0)
        results.append(check_gate(gate, value))

    return results


def check_calibration_gates(
    calibration_metrics: Dict, initial_ece: float = 0.0
) -> List[GateResult]:
    """Check calibration improvement gates.

    Args:
        calibration_metrics: Calibration metrics with improvement_pct
        initial_ece: Initial ECE before calibration

    Returns:
        List of GateResult objects
    """
    results = []

    # Only check improvement if initial ECE > 0.05
    if initial_ece > 0.05:
        improvement_pct = calibration_metrics.get("improvement_pct", 0.0)
        results.append(check_gate(CALIBRATION_GATES[0], improvement_pct))

    return results


def check_data_leakage(
    train_post_ids: List[str],
    test_post_ids: List[str],
    fold: int,
) -> GateResult:
    """Check for data leakage between train and test sets.

    Args:
        train_post_ids: List of post IDs in training set
        test_post_ids: List of post IDs in test set
        fold: Fold number

    Returns:
        GateResult indicating if leakage was detected
    """
    train_set = set(train_post_ids)
    test_set = set(test_post_ids)

    overlap = train_set & test_set
    passed = len(overlap) == 0

    if passed:
        message = f"Fold {fold}: No data leakage detected (PASS)"
    else:
        message = (
            f"Fold {fold}: Data leakage detected! {len(overlap)} posts "
            f"in both train and test (FAIL)"
        )

    gate = QualityGate(
        name=f"fold_{fold}_no_leakage",
        metric="leakage",
        threshold=0.0,
        comparison="lte",
        description="No post overlap between train and test",
    )

    return GateResult(gate=gate, value=float(len(overlap)), passed=passed, message=message)


def run_acceptance_checks(
    retrieval_metrics_path: Path,
    evidence_metrics_path: Path,
    fold_info_path: Path | None = None,
    output_path: Path | None = None,
    strict: bool = True,
) -> Dict:
    """Run all acceptance checks and generate verdict.

    Args:
        retrieval_metrics_path: Path to retrieval metrics JSON
        evidence_metrics_path: Path to evidence metrics JSON
        fold_info_path: Optional path to fold info with train/test splits
        output_path: Optional path to save acceptance report
        strict: If True, raise QualityGateError on failure

    Returns:
        Dict with acceptance report

    Raises:
        QualityGateError: If strict=True and any gates fail
    """
    logger = get_logger(__name__)

    # Load metrics
    with open(retrieval_metrics_path) as f:
        retrieval_metrics = json.load(f)

    with open(evidence_metrics_path) as f:
        evidence_metrics = json.load(f)

    # Run quality gate checks
    retrieval_results = check_retrieval_gates(retrieval_metrics)
    evidence_results = check_evidence_gates(evidence_metrics)

    # Check for data leakage if fold info provided
    leakage_results = []
    if fold_info_path and fold_info_path.exists():
        with open(fold_info_path) as f:
            fold_info = json.load(f)

        for fold_data in fold_info.get("folds", []):
            fold = fold_data["fold"]
            train_posts = fold_data.get("train_post_ids", [])
            test_posts = fold_data.get("test_post_ids", [])
            leakage_results.append(check_data_leakage(train_posts, test_posts, fold))

    # Collect all results
    all_results = retrieval_results + evidence_results + leakage_results

    # Generate report
    n_passed = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    all_passed = n_passed == n_total

    report = {
        "verdict": "ACCEPTED" if all_passed else "FAILED",
        "n_passed": n_passed,
        "n_total": n_total,
        "pass_rate": n_passed / n_total if n_total > 0 else 0.0,
        "results": [
            {
                "gate": r.gate.name,
                "metric": r.gate.metric,
                "threshold": r.gate.threshold,
                "value": r.value,
                "passed": r.passed,
                "message": r.message,
            }
            for r in all_results
        ],
    }

    # Log results
    logger.info("=" * 60)
    logger.info("ACCEPTANCE CHECK RESULTS")
    logger.info("=" * 60)
    for result in all_results:
        logger.info(result.message)
    logger.info("=" * 60)
    logger.info(f"VERDICT: {report['verdict']} ({n_passed}/{n_total} passed)")
    logger.info("=" * 60)

    # Save report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved acceptance report to {output_path}")

    # Raise error if strict mode and failed
    if strict and not all_passed:
        failed_gates = [r for r in all_results if not r.passed]
        error_msg = (
            f"Quality gates FAILED: {len(failed_gates)} of {n_total} gates did not pass\n"
            + "\n".join(f"  - {r.message}" for r in failed_gates)
        )
        raise QualityGateError(error_msg)

    return report
