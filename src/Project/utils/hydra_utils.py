"""Hydra/OmegaConf helpers used by scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(cfg_path: str | Path, validate: bool = False) -> DictConfig:
    """Load configuration from YAML file with optional validation.

    Args:
        cfg_path: Path to config file
        validate: If True, validate key sections (train, dataloader) using Pydantic schemas

    Returns:
        Loaded configuration

    Raises:
        ValidationError: If validate=True and config is invalid
    """
    cfg = OmegaConf.load(str(cfg_path))

    if validate:
        from Project.utils.config_schema import TrainConfig, DataLoaderConfig, validate_config
        from pydantic import ValidationError
        from Project.utils.logging import get_logger

        logger = get_logger(__name__)

        # Validate train section if present
        if "train" in cfg:
            try:
                validate_config(cfg.train, TrainConfig)
                logger.info("✓ Train config validation passed")
            except ValidationError as e:
                logger.error("Train config validation failed: %s", e)
                raise

        # Validate dataloader section if present
        if "dataloader" in cfg:
            try:
                validate_config(cfg.dataloader, DataLoaderConfig)
                logger.info("✓ DataLoader config validation passed")
            except ValidationError as e:
                logger.error("DataLoader config validation failed: %s", e)
                raise

    return cfg


def save_config(cfg: DictConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = output_dir / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, cfg_dir / "config_resolved.yaml")


def cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Safely get nested config value with dot notation.

    Args:
        cfg: OmegaConf config object or dict
        path: Dot-separated key path (e.g., 'model.encoder.hidden_size')
        default: Default value if key not found

    Returns:
        Config value or default

    Examples:
        >>> cfg = OmegaConf.create({"model": {"lr": 0.001}})
        >>> cfg_get(cfg, "model.lr", 0.01)
        0.001
        >>> cfg_get(cfg, "model.dropout", 0.1)
        0.1
    """
    try:
        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        # Fallback for non-OmegaConf dicts
        node = cfg
        for key in path.split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            elif hasattr(node, key):
                node = getattr(node, key)
            else:
                return default
        return node
