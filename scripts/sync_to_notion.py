#!/usr/bin/env python3
"""Sync training and HPO results to Notion.

This script syncs experiment results from Optuna (HPO) and training outputs to Notion databases.

Environment Variables Required:
    NOTION_API_TOKEN: Your Notion integration token
    NOTION_DATABASE_ID: ID of the Notion database for HPO/experiment results

Optional Environment Variables:
    NOTION_TRAINING_DB_ID: Separate database ID for training runs (uses NOTION_DATABASE_ID if not set)

Usage:
    # Sync all results (HPO studies + training runs)
    python scripts/sync_to_notion.py
    
    # Sync specific HPO study
    python scripts/sync_to_notion.py --study real_dev_hpo_refine_hpo
    
    # Sync specific training run
    python scripts/sync_to_notion.py --run outputs/runs/real_dev_hpo_refine_trial_8
    
    # Sync HPO summary report
    python scripts/sync_to_notion.py --summary outputs/real_dev_hpo_refine_final_report.json
    
    # Filter by pattern (only sync items containing "real_dev")
    python scripts/sync_to_notion.py --pattern real_dev
    
    # Sync only best trials from HPO studies
    python scripts/sync_to_notion.py --best-only

Example Notion Database Setup:
    Create a Notion database with these properties:
    - Study (Title)
    - Trial (Text)
    - Status (Select: COMPLETE, RUNNING, PRUNED, FAILED)
    - Objective Value (Number)
    - F1 (Number)
    - Macro F1 (Number)
    - Precision@5 (Number)
    - ECE (Number)
    - Learning Rate (Number)
    - Batch Size (Number)
    - Epochs (Number)
    - LoRA Rank (Number)
    - LoRA Alpha (Number)
    - LoRA Dropout (Number)
    - Neg Per Pos (Number)
    - Rank Weight (Number)
    - Rank Margin (Number)
    - Is Best (Checkbox)
    - Start Time (Date)
    - End Time (Date)
    - Parameters JSON (Text)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from Project.utils.notion_utils import NotionSync, sync_all_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_env_vars() -> bool:
    """Check if required environment variables are set."""
    token = os.environ.get("NOTION_API_TOKEN")
    db_id = os.environ.get("NOTION_DATABASE_ID")

    if not token:
        logger.error(
            "NOTION_API_TOKEN not set. "
            "Get your token from: https://www.notion.so/my-integrations"
        )
        return False

    if not db_id:
        logger.error(
            "NOTION_DATABASE_ID not set. "
            "Get the database ID from the Notion database URL: "
            "https://www.notion.so/<workspace>/<database_id>?v=..."
        )
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync experiment results to Notion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--study",
        type=str,
        help="Sync specific Optuna study by name",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Sync specific training run directory",
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="Sync specific HPO summary JSON file",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Filter items by pattern (e.g., 'real_dev')",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="For HPO studies, only sync the best trial",
    )
    parser.add_argument(
        "--optuna-db",
        type=str,
        default="sqlite:///optuna.db",
        help="Path to Optuna database (default: sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Path to outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without actually syncing",
    )
    args = parser.parse_args()

    # Check environment variables
    if not args.dry_run and not check_env_vars():
        sys.exit(1)

    # Change to project root
    os.chdir(project_root)
    logger.info(f"Working directory: {os.getcwd()}")

    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("Would sync the following:")

        # Show what would be synced
        outputs_path = Path(args.outputs_dir)

        # HPO studies
        db_path = args.optuna_db.replace("sqlite:///", "")
        if Path(db_path).exists():
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT study_name FROM studies")
            studies = [row[0] for row in cursor.fetchall()]
            conn.close()

            for study in studies:
                if args.pattern and args.pattern not in study:
                    continue
                if args.study and args.study != study:
                    continue
                logger.info(f"  [HPO] Study: {study}")

        # Training runs
        runs_dir = outputs_path / "runs"
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if args.pattern and args.pattern not in run_dir.name:
                    continue
                if args.run and args.run != str(run_dir):
                    continue
                if (run_dir / "evidence_metrics.json").exists():
                    logger.info(f"  [Training] Run: {run_dir.name}")

        # Summary files
        for summary_file in sorted(outputs_path.glob("*_final_report.json")):
            if args.pattern and args.pattern not in summary_file.name:
                continue
            if args.summary and args.summary != str(summary_file):
                continue
            logger.info(f"  [Summary] {summary_file.name}")

        logger.info("\nTo actually sync, remove --dry-run flag")
        return

    try:
        sync = NotionSync()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    results = {"hpo_pages": [], "training_pages": [], "summary_pages": []}

    # Sync specific study
    if args.study:
        logger.info(f"Syncing HPO study: {args.study}")
        pages = sync.sync_hpo_study(
            args.study,
            args.optuna_db,
            include_all_trials=not args.best_only,
        )
        results["hpo_pages"].extend(pages)

    # Sync specific run
    elif args.run:
        logger.info(f"Syncing training run: {args.run}")
        page_id = sync.sync_training_run(args.run)
        if page_id:
            results["training_pages"].append(page_id)

    # Sync specific summary
    elif args.summary:
        logger.info(f"Syncing HPO summary: {args.summary}")
        page_id = sync.sync_hpo_summary(args.summary)
        if page_id:
            results["summary_pages"].append(page_id)

    # Sync all
    else:
        logger.info("Syncing all results to Notion...")
        results = sync_all_results(
            optuna_db=args.optuna_db,
            outputs_dir=args.outputs_dir,
            study_pattern=args.pattern,
        )

    # Report results
    logger.info("=" * 60)
    logger.info("Sync Complete!")
    logger.info(f"  HPO trial pages created: {len(results['hpo_pages'])}")
    logger.info(f"  Training run pages created: {len(results['training_pages'])}")
    logger.info(f"  Summary pages created: {len(results['summary_pages'])}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

