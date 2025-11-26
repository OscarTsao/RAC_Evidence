"""Run acceptance checks on Evidence RAC pipeline results.

Usage:
    python scripts/run_acceptance_checks.py --exp real_dev
    python scripts/run_acceptance_checks.py --exp real_dev --strict  # Fail on gate violations
"""

import argparse
from pathlib import Path

from Project.metrics import run_acceptance_checks
from Project.utils.logging import get_logger


def main(exp: str, strict: bool = False) -> None:
    """Run acceptance checks for an experiment.

    Args:
        exp: Experiment name
        strict: If True, raise error on quality gate failures
    """
    logger = get_logger(__name__)

    # Define paths
    base_dir = Path(f"outputs/runs/{exp}")
    retrieval_metrics_path = base_dir / "retrieval_sweep" / "sweep_metrics.json"
    evidence_metrics_path = base_dir / "evidence_metrics.json"
    fold_info_path = base_dir / "fold_info.json"
    output_path = base_dir / "acceptance_report.json"

    # Check that required files exist
    if not retrieval_metrics_path.exists():
        logger.error(f"Retrieval metrics not found: {retrieval_metrics_path}")
        logger.error("Run retrieval sweep first: make sweep_recall EXP=%s", exp)
        return

    if not evidence_metrics_path.exists():
        logger.error(f"Evidence metrics not found: {evidence_metrics_path}")
        logger.error("Run evidence training first: make train_evidence_5fold EXP=%s", exp)
        return

    logger.info(f"Running acceptance checks for experiment: {exp}")
    logger.info(f"Retrieval metrics: {retrieval_metrics_path}")
    logger.info(f"Evidence metrics: {evidence_metrics_path}")

    try:
        report = run_acceptance_checks(
            retrieval_metrics_path=retrieval_metrics_path,
            evidence_metrics_path=evidence_metrics_path,
            fold_info_path=fold_info_path if fold_info_path.exists() else None,
            output_path=output_path,
            strict=strict,
        )

        logger.info("")
        logger.info(f"Acceptance report saved to: {output_path}")
        logger.info(f"Final verdict: {report['verdict']}")
        logger.info(f"Gates passed: {report['n_passed']}/{report['n_total']}")

        if report["verdict"] == "ACCEPTED":
            logger.info("✓ All quality gates passed!")
        else:
            logger.warning("✗ Some quality gates failed")

    except Exception as e:
        logger.error(f"Acceptance checks failed: {e}")
        if strict:
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run acceptance checks on Evidence RAC pipeline results"
    )
    parser.add_argument(
        "--exp", required=True, help="Experiment name (e.g., real_dev)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with error if quality gates don't pass",
    )

    args = parser.parse_args()
    main(args.exp, args.strict)
