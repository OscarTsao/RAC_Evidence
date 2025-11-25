"""Optuna hyperparameter optimization utilities."""

from __future__ import annotations

import logging
from typing import Any

import optuna
from omegaconf import DictConfig
from optuna.pruners import MedianPruner

logger = logging.getLogger(__name__)


def create_study(
    study_name: str,
    storage_uri: str = "sqlite:///optuna.db",
    direction: str = "maximize",
    pruner: optuna.pruners.BasePruner | None = None,
) -> optuna.Study:
    """Create or load Optuna study.

    Args:
        study_name: Name for the study
        storage_uri: SQLite database path
        direction: 'maximize' or 'minimize'
        pruner: Pruner for early stopping

    Returns:
        Optuna study object

    Examples:
        >>> study = create_study("my_experiment", direction="maximize")
        >>> # Run optimization
        >>> study.optimize(objective, n_trials=50)
    """
    if pruner is None:
        pruner = MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
        direction=direction,
        pruner=pruner,
    )

    logger.info(f"Created/loaded Optuna study: {study_name}")
    return study


def suggest_hyperparameters(trial: optuna.Trial, cfg: DictConfig) -> dict[str, Any]:
    """Suggest hyperparameters for a trial.

    Args:
        trial: Optuna trial object
        cfg: Base configuration

    Returns:
        Dict of suggested hyperparameters

    Examples:
        >>> import optuna
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"model": {"encoder": "bert"}})
        >>> def objective(trial):
        ...     params = suggest_hyperparameters(trial, cfg)
        ...     # Train model with params
        ...     return val_score
    """
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }

    # Add model-specific params based on config
    if "encoder" in cfg.get("model", {}):
        params["encoder_lr"] = trial.suggest_float("encoder_lr", 1e-6, 1e-4, log=True)

    if "hidden" in cfg.get("model", {}):
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])

    return params


def log_trial_params(trial: optuna.Trial, params: dict[str, Any]) -> None:
    """Log trial parameters (useful for MLflow integration).

    Args:
        trial: Optuna trial object
        params: Parameters to log

    Examples:
        >>> trial = study.ask()
        >>> params = {"lr": 0.001, "batch_size": 32}
        >>> log_trial_params(trial, params)
    """
    for key, value in params.items():
        trial.set_user_attr(key, value)
