"""Notion API utilities for syncing experiment results.

This module provides utilities to write training and HPO results to Notion databases.
Requires the following environment variables:
    - NOTION_API_TOKEN: Notion integration token
    - NOTION_DATABASE_ID: ID of the Notion database to write to (for HPO runs)
    - NOTION_TRAINING_DB_ID: ID of the Notion database for training runs (optional)

Example usage:
    from Project.utils.notion_utils import NotionSync
    
    sync = NotionSync()
    sync.sync_hpo_study("real_dev_hpo_refine_hpo")
    sync.sync_training_run("outputs/runs/real_dev_hpo_refine_trial_8")
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_env_or_raise(key: str) -> str:
    """Get environment variable or raise an error."""
    value = os.environ.get(key)
    if not value:
        raise ValueError(
            f"Environment variable {key} is required. "
            f"Please set it before using Notion integration."
        )
    return value


class NotionSync:
    """Sync experiment results to Notion databases.
    
    Attributes:
        client: Notion API client
        hpo_database_id: Database ID for HPO trials
        training_database_id: Database ID for training runs
    """

    def __init__(
        self,
        api_token: str | None = None,
        hpo_database_id: str | None = None,
        training_database_id: str | None = None,
    ):
        """Initialize Notion sync client.
        
        Args:
            api_token: Notion API token (defaults to NOTION_API_TOKEN env var)
            hpo_database_id: Database ID for HPO runs (defaults to NOTION_DATABASE_ID env var)
            training_database_id: Database ID for training runs (defaults to NOTION_TRAINING_DB_ID env var)
        """
        try:
            from notion_client import Client
        except ImportError:
            raise ImportError(
                "notion-client is required. Install with: pip install notion-client"
            )

        self.api_token = api_token or _get_env_or_raise("NOTION_API_TOKEN")
        self.hpo_database_id = hpo_database_id or os.environ.get("NOTION_DATABASE_ID")
        self.training_database_id = training_database_id or os.environ.get(
            "NOTION_TRAINING_DB_ID"
        )

        self.client = Client(auth=self.api_token)
        logger.info("Notion client initialized successfully")

    def _create_page_properties(
        self, data: dict[str, Any], schema: dict[str, str]
    ) -> dict[str, Any]:
        """Convert data dict to Notion page properties format.
        
        Args:
            data: Dictionary of field values
            schema: Mapping of field names to Notion property types
                    Supported types: title, rich_text, number, select, date, checkbox
        
        Returns:
            Notion properties dict
        """
        properties = {}
        for key, prop_type in schema.items():
            value = data.get(key)
            if value is None:
                continue

            if prop_type == "title":
                properties[key] = {"title": [{"text": {"content": str(value)}}]}
            elif prop_type == "rich_text":
                properties[key] = {"rich_text": [{"text": {"content": str(value)}}]}
            elif prop_type == "number":
                properties[key] = {"number": float(value) if value is not None else None}
            elif prop_type == "select":
                properties[key] = {"select": {"name": str(value)}}
            elif prop_type == "date":
                if isinstance(value, datetime):
                    properties[key] = {"date": {"start": value.isoformat()}}
                elif isinstance(value, str):
                    properties[key] = {"date": {"start": value}}
            elif prop_type == "checkbox":
                properties[key] = {"checkbox": bool(value)}
            elif prop_type == "multi_select":
                if isinstance(value, list):
                    properties[key] = {"multi_select": [{"name": str(v)} for v in value]}
                else:
                    properties[key] = {"multi_select": [{"name": str(value)}]}

        return properties

    def sync_hpo_study(
        self,
        study_name: str,
        optuna_db_path: str = "sqlite:///optuna.db",
        include_all_trials: bool = True,
    ) -> list[str]:
        """Sync an Optuna HPO study to Notion.
        
        Args:
            study_name: Name of the Optuna study
            optuna_db_path: Path to Optuna SQLite database
            include_all_trials: If True, sync all trials; if False, only sync best trial
        
        Returns:
            List of created Notion page IDs
        """
        if not self.hpo_database_id:
            raise ValueError(
                "HPO database ID not set. Set NOTION_DATABASE_ID env var or pass hpo_database_id."
            )

        # Parse database path
        if optuna_db_path.startswith("sqlite:///"):
            db_path = optuna_db_path.replace("sqlite:///", "")
        else:
            db_path = optuna_db_path

        if not Path(db_path).exists():
            raise FileNotFoundError(f"Optuna database not found: {db_path}")

        # Load study data from SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study ID
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Study '{study_name}' not found in database")
        study_id = result[0]

        # Get trials
        cursor.execute(
            """
            SELECT t.trial_id, t.number, t.state, t.datetime_start, t.datetime_complete,
                   tv.value
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            """,
            (study_id,),
        )
        trials = cursor.fetchall()

        if not trials:
            logger.warning(f"No completed trials found for study '{study_name}'")
            conn.close()
            return []

        created_pages = []
        trials_to_sync = trials if include_all_trials else [trials[0]]  # Best is first

        for trial in trials_to_sync:
            trial_id, number, state, start_time, end_time, value = trial

            # Get trial parameters
            cursor.execute(
                "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?",
                (trial_id,),
            )
            params = {row[0]: row[1] for row in cursor.fetchall()}

            # Get trial user attributes (metrics)
            cursor.execute(
                "SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?",
                (trial_id,),
            )
            user_attrs = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

            # Prepare data for Notion
            trial_data = {
                "Study": study_name,
                "Trial": f"Trial {number}",
                "Status": state,
                "Objective Value": value,
                "Start Time": start_time,
                "End Time": end_time,
                "Is Best": number == trials[0][1],  # First trial is best (sorted by value DESC)
                # Common HPO parameters
                "Learning Rate": params.get("lr"),
                "Batch Size": params.get("batch_size"),
                "Epochs": params.get("epochs"),
                "LoRA Rank": params.get("lora_r"),
                "LoRA Alpha": params.get("lora_alpha"),
                "LoRA Dropout": params.get("lora_dropout"),
                "Neg Per Pos": params.get("neg_per_pos"),
                "Rank Weight": params.get("rank_weight"),
                "Rank Margin": params.get("rank_margin"),
                # Metrics from user attributes
                "F1": user_attrs.get("f1"),
                "Macro F1": user_attrs.get("macro_f1"),
                "Precision@5": user_attrs.get("precision@5"),
                "ECE": user_attrs.get("ece"),
                # Store full params as JSON string
                "Parameters JSON": json.dumps(params, indent=2),
            }

            schema = {
                "Study": "title",
                "Trial": "rich_text",
                "Status": "select",
                "Objective Value": "number",
                "Start Time": "date",
                "End Time": "date",
                "Is Best": "checkbox",
                "Learning Rate": "number",
                "Batch Size": "number",
                "Epochs": "number",
                "LoRA Rank": "number",
                "LoRA Alpha": "number",
                "LoRA Dropout": "number",
                "Neg Per Pos": "number",
                "Rank Weight": "number",
                "Rank Margin": "number",
                "F1": "number",
                "Macro F1": "number",
                "Precision@5": "number",
                "ECE": "number",
                "Parameters JSON": "rich_text",
            }

            properties = self._create_page_properties(trial_data, schema)

            try:
                response = self.client.pages.create(
                    parent={"database_id": self.hpo_database_id},
                    properties=properties,
                )
                page_id = response["id"]
                created_pages.append(page_id)
                logger.info(f"Created Notion page for Trial {number}: {page_id}")
            except Exception as e:
                logger.error(f"Failed to create page for Trial {number}: {e}")

        conn.close()
        logger.info(f"Synced {len(created_pages)} trials to Notion")
        return created_pages

    def sync_training_run(
        self,
        run_dir: str | Path,
        run_name: str | None = None,
        database_id: str | None = None,
    ) -> str | None:
        """Sync a training run's metrics to Notion.
        
        Args:
            run_dir: Path to the training run output directory
            run_name: Optional name for the run (defaults to directory name)
            database_id: Override database ID (defaults to training_database_id)
        
        Returns:
            Created Notion page ID or None if failed
        """
        db_id = database_id or self.training_database_id or self.hpo_database_id
        if not db_id:
            raise ValueError("No database ID available for training runs")

        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        run_name = run_name or run_dir.name

        # Load metrics files
        metrics_data = {}

        # Try loading evidence_metrics.json
        evidence_metrics_path = run_dir / "evidence_metrics.json"
        if evidence_metrics_path.exists():
            with open(evidence_metrics_path) as f:
                evidence_metrics = json.load(f)
            metrics_data["evidence_metrics"] = evidence_metrics

        # Try loading fold_temperatures.json (calibration info)
        temperatures_path = run_dir / "fold_temperatures.json"
        if temperatures_path.exists():
            with open(temperatures_path) as f:
                temperatures = json.load(f)
            metrics_data["calibration"] = temperatures

        # Get overall metrics
        overall = metrics_data.get("evidence_metrics", {}).get("overall", {})

        # Prepare data for Notion
        run_data = {
            "Run Name": run_name,
            "Status": "COMPLETE",
            "Created": datetime.now().isoformat(),
            # Core metrics
            "F1": overall.get("f1"),
            "Macro F1": overall.get("macro_f1"),
            "Micro F1": overall.get("micro_f1"),
            "AUROC": overall.get("auroc"),
            "AUPRC": overall.get("auprc"),
            "Precision": overall.get("precision"),
            "Precision@5": overall.get("precision@5"),
            "Precision@10": overall.get("precision@10"),
            "Recall@5": overall.get("recall@5"),
            "Recall@10": overall.get("recall@10"),
            "ECE": overall.get("ece"),
            # Calibration improvement
            "ECE Before Cal": metrics_data.get("evidence_metrics", {})
            .get("calibration_improvement", {})
            .get("ece_before"),
            "ECE After Cal": metrics_data.get("evidence_metrics", {})
            .get("calibration_improvement", {})
            .get("ece_after"),
            # Store full metrics as JSON
            "Metrics JSON": json.dumps(overall, indent=2) if overall else None,
        }

        schema = {
            "Run Name": "title",
            "Status": "select",
            "Created": "date",
            "F1": "number",
            "Macro F1": "number",
            "Micro F1": "number",
            "AUROC": "number",
            "AUPRC": "number",
            "Precision": "number",
            "Precision@5": "number",
            "Precision@10": "number",
            "Recall@5": "number",
            "Recall@10": "number",
            "ECE": "number",
            "ECE Before Cal": "number",
            "ECE After Cal": "number",
            "Metrics JSON": "rich_text",
        }

        properties = self._create_page_properties(run_data, schema)

        try:
            response = self.client.pages.create(
                parent={"database_id": db_id},
                properties=properties,
            )
            page_id = response["id"]
            logger.info(f"Created Notion page for run '{run_name}': {page_id}")
            return page_id
        except Exception as e:
            logger.error(f"Failed to create page for run '{run_name}': {e}")
            return None

    def sync_hpo_summary(
        self,
        summary_path: str | Path,
        study_name: str | None = None,
        database_id: str | None = None,
    ) -> str | None:
        """Sync an HPO summary report (JSON) to Notion.
        
        Args:
            summary_path: Path to the HPO summary JSON file
            study_name: Optional study name (extracted from file if not provided)
            database_id: Override database ID
        
        Returns:
            Created Notion page ID or None if failed
        """
        db_id = database_id or self.hpo_database_id
        if not db_id:
            raise ValueError("No database ID available")

        summary_path = Path(summary_path)
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")

        with open(summary_path) as f:
            summary = json.load(f)

        study_name = study_name or summary_path.stem

        # Extract data from summary
        phase_summary = summary.get("phase3_summary", summary.get("summary", {}))
        best_params = summary.get("best_hyperparameters", {})
        best_metrics = summary.get("best_metrics", {})
        improvement = summary.get("improvement", {})
        baseline = summary.get("phase2_baseline", {})

        summary_data = {
            "Study": f"{study_name} - Summary",
            "Trial": f"Best: Trial {phase_summary.get('best_trial', 'N/A')}",
            "Status": summary.get("conclusion", "COMPLETE"),
            "Total Trials": phase_summary.get("total_trials"),
            "Successful Trials": phase_summary.get("successful_trials"),
            # Best metrics
            "F1": best_metrics.get("f1"),
            "Macro F1": best_metrics.get("macro_f1"),
            "Precision@5": best_metrics.get("precision@5"),
            "ECE": best_metrics.get("ece"),
            # Best hyperparameters
            "Learning Rate": best_params.get("lr"),
            "Rank Weight": best_params.get("rank_weight"),
            "Rank Margin": best_params.get("rank_margin"),
            "LoRA Dropout": best_params.get("lora_dropout"),
            "Neg Per Pos": best_params.get("neg_per_pos"),
            # Improvement over baseline
            "F1 Improvement (%)": improvement.get("f1_pct"),
            "Macro F1 Improvement (%)": improvement.get("macro_f1_pct"),
            "Is Best": True,
            # Full summary as JSON
            "Parameters JSON": json.dumps(summary, indent=2),
        }

        schema = {
            "Study": "title",
            "Trial": "rich_text",
            "Status": "select",
            "Total Trials": "number",
            "Successful Trials": "number",
            "F1": "number",
            "Macro F1": "number",
            "Precision@5": "number",
            "ECE": "number",
            "Learning Rate": "number",
            "Rank Weight": "number",
            "Rank Margin": "number",
            "LoRA Dropout": "number",
            "Neg Per Pos": "number",
            "F1 Improvement (%)": "number",
            "Macro F1 Improvement (%)": "number",
            "Is Best": "checkbox",
            "Parameters JSON": "rich_text",
        }

        properties = self._create_page_properties(summary_data, schema)

        try:
            response = self.client.pages.create(
                parent={"database_id": db_id},
                properties=properties,
            )
            page_id = response["id"]
            logger.info(f"Created Notion page for summary '{study_name}': {page_id}")
            return page_id
        except Exception as e:
            logger.error(f"Failed to create summary page: {e}")
            return None


def sync_all_results(
    optuna_db: str = "sqlite:///optuna.db",
    outputs_dir: str = "outputs",
    study_pattern: str | None = None,
) -> dict[str, list[str]]:
    """Convenience function to sync all HPO and training results to Notion.
    
    Args:
        optuna_db: Path to Optuna database
        outputs_dir: Path to outputs directory
        study_pattern: Optional pattern to filter studies (e.g., "real_dev")
    
    Returns:
        Dict with keys 'hpo_pages', 'training_pages', 'summary_pages' containing created page IDs
    """
    sync = NotionSync()
    results = {"hpo_pages": [], "training_pages": [], "summary_pages": []}
    outputs_path = Path(outputs_dir)

    # Sync HPO studies
    if Path(optuna_db.replace("sqlite:///", "")).exists():
        conn = sqlite3.connect(optuna_db.replace("sqlite:///", ""))
        cursor = conn.cursor()
        cursor.execute("SELECT study_name FROM studies")
        studies = [row[0] for row in cursor.fetchall()]
        conn.close()

        for study_name in studies:
            if study_pattern and study_pattern not in study_name:
                continue
            try:
                pages = sync.sync_hpo_study(study_name, optuna_db)
                results["hpo_pages"].extend(pages)
            except Exception as e:
                logger.error(f"Failed to sync study '{study_name}': {e}")

    # Sync training runs
    runs_dir = outputs_path / "runs"
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if study_pattern and study_pattern not in run_dir.name:
                continue
            # Only sync if it has evidence_metrics.json
            if (run_dir / "evidence_metrics.json").exists():
                try:
                    page_id = sync.sync_training_run(run_dir)
                    if page_id:
                        results["training_pages"].append(page_id)
                except Exception as e:
                    logger.error(f"Failed to sync run '{run_dir.name}': {e}")

    # Sync summary files
    for summary_file in outputs_path.glob("*_final_report.json"):
        if study_pattern and study_pattern not in summary_file.name:
            continue
        try:
            page_id = sync.sync_hpo_summary(summary_file)
            if page_id:
                results["summary_pages"].append(page_id)
        except Exception as e:
            logger.error(f"Failed to sync summary '{summary_file.name}': {e}")

    total = sum(len(v) for v in results.values())
    logger.info(f"Synced {total} pages to Notion")
    return results

