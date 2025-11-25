"""MLflow experiment tracking utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def setup_mlflow(cfg: DictConfig) -> str:
    """Configure MLflow experiment.

    Args:
        cfg: Configuration with mlflow settings

    Returns:
        Experiment ID

    Examples:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"mlflow_tracking_uri": "file:./mlruns", "experiment_name": "demo"})
        >>> exp_id = setup_mlflow(cfg)
    """
    tracking_uri = cfg.get("mlflow_tracking_uri", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = cfg.get("experiment_name", "default")
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created MLflow experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {experiment_name}")

    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


@contextmanager
def mlflow_run(run_name: str, cfg: DictConfig, nested: bool = False):
    """Context manager for MLflow run.

    Args:
        run_name: Name for this run
        cfg: Configuration to log
        nested: Whether this is a nested run (for HPO)

    Yields:
        MLflow run object

    Examples:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"exp": "test", "model": {"encoder": "bert"}})
        >>> with mlflow_run("training", cfg):
        ...     # Training code here
        ...     log_metrics({"loss": 0.1})
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        # Log all config parameters
        params_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(params_dict, dict):
            mlflow.log_params(_flatten_dict(params_dict))

        # Set tags
        mlflow.set_tags({
            "experiment": cfg.get("exp", "unknown"),
            "model": str(cfg.get("model", {}).get("encoder", "unknown")),
        })

        yield run


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dictionary for MLflow logging.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary

    Examples:
        >>> _flatten_dict({"a": {"b": 1}, "c": 2})
        {'a.b': 1, 'c': 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to current MLflow run.

    Args:
        metrics: Dict of metric name -> value
        step: Optional step/epoch number

    Examples:
        >>> log_metrics({"train_loss": 0.5, "val_loss": 0.6}, step=10)
    """
    mlflow.log_metrics(metrics, step=step)


def log_artifacts(output_dir: Path) -> None:
    """Log all files in output directory as artifacts.

    Args:
        output_dir: Directory containing artifacts

    Examples:
        >>> from pathlib import Path
        >>> log_artifacts(Path("outputs/checkpoints"))
    """
    if output_dir.exists():
        mlflow.log_artifacts(str(output_dir))
