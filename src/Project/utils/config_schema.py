"""Pydantic validation schemas for all configuration files.

This module provides type-safe validation for YAML configuration files used
throughout the project. All config files should be validated against these
schemas to catch errors early.

Example usage:
    from omegaconf import OmegaConf
    from Project.utils.config_schema import validate_config, TrainConfig

    cfg = OmegaConf.load("configs/bi.yaml")
    train_cfg = validate_config(cfg.train, TrainConfig)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Data Configuration Schemas
# ============================================================================


class DataConfig(BaseModel):
    """Configuration for data paths and processing parameters."""

    raw_dir: str = Field(default="data/raw")
    processed_dir: str = Field(default="data/processed")
    interim_dir: str = Field(default="data/interim")
    chunk_max_len: Optional[int] = Field(default=512, gt=0)
    stride: Optional[int] = Field(default=128, ge=0)
    sent_max_len: Optional[int] = Field(default=128, gt=0)


class SplitConfig(BaseModel):
    """Configuration for train/val/test splits."""

    seed: int = Field(default=42)


# ============================================================================
# Training Configuration Schemas
# ============================================================================


class TrainConfig(BaseModel):
    """Configuration for training loop parameters."""

    lr: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    use_amp: bool = Field(default=False)
    amp_dtype: Literal["bf16", "fp16", "fp32"] = Field(default="bf16")
    grad_accum: Optional[int] = Field(default=1, gt=0)
    shuffle: Optional[bool] = Field(default=True)
    temperature: Optional[float] = Field(default=None, gt=0)
    batch_size_posts: Optional[int] = Field(default=None, gt=0)

    @field_validator("lr")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Learning rate must be positive, got {v}")
        return v

    @field_validator("batch_size", "epochs")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class DataLoaderConfig(BaseModel):
    """Configuration for PyTorch DataLoader."""

    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = Field(default=False)
    persistent_workers: bool = Field(default=False)
    prefetch_factor: Optional[int] = Field(default=2, ge=1)

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"num_workers must be >= 0, got {v}")
        return v


# ============================================================================
# Retrieval Configuration Schemas
# ============================================================================


class RetrieveConfig(BaseModel):
    """Configuration for basic retrieval."""

    per_post: Optional[bool] = Field(default=True)
    topK: int = Field(default=50, gt=0)

    @field_validator("topK")
    @classmethod
    def validate_topK(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"topK must be positive, got {v}")
        return v


class RerankConfig(BaseModel):
    """Configuration for reranking."""

    keep_topk: int = Field(default=10, gt=0)
    agg: Literal["max", "topm", "logsumexp"] = Field(default="max")
    topm: Optional[int] = Field(default=2, gt=0)


# ============================================================================
# Loss Configuration Schemas
# ============================================================================


class LossConfig(BaseModel):
    """Configuration for loss function parameters."""

    bce_pos_weight: Optional[float] = Field(default=None, gt=0)
    ranking_gamma: Optional[float] = Field(default=None, gt=0)
    ranking_margin: Optional[float] = Field(default=None, ge=0)
    lambda_edge: Optional[float] = Field(default=None, ge=0)
    lambda_node: Optional[float] = Field(default=None, ge=0)
    lambda_cons: Optional[float] = Field(default=None, ge=0)
    cons_margin: Optional[float] = Field(default=None, ge=0)


# ============================================================================
# Model Configuration Schemas
# ============================================================================


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    name: Optional[str] = Field(default=None)
    type: Optional[Literal["HGT", "GAT", "GCN"]] = Field(default=None)
    layers: Optional[int] = Field(default=None, gt=0)
    hidden: Optional[int] = Field(default=None, gt=0)
    dropout: Optional[float] = Field(default=0.1, ge=0.0, le=1.0)


class GraphNodeFeatsConfig(BaseModel):
    """Configuration for graph node features."""

    type: str
    dim: int = Field(gt=0)
    normalize: Optional[bool] = Field(default=False)
    add_is_core: Optional[bool] = Field(default=False)


class GraphEdgeFeatsConfig(BaseModel):
    """Configuration for graph edge features."""

    feat: List[str]
    topk_per_cid: Optional[int] = Field(default=None, gt=0)


class GraphEdgesConfig(BaseModel):
    """Configuration for graph edge construction."""

    strategy: Optional[str] = Field(default=None)
    k: Optional[int] = Field(default=None, gt=0)
    source: Optional[str] = Field(default=None)
    feat: Optional[List[str]] = Field(default=None)


class GraphConfig(BaseModel):
    """Configuration for graph structure."""

    node_feats: Optional[Dict[str, Any]] = Field(default=None)
    edges: Optional[Dict[str, Any]] = Field(default=None)


# ============================================================================
# Validation Helper
# ============================================================================


def validate_config(cfg: DictConfig | dict | Any, schema_cls: type[BaseModel]) -> BaseModel:
    """Validate a configuration against a Pydantic schema.

    Args:
        cfg: Configuration object (from OmegaConf or dict)
        schema_cls: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If configuration is invalid

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("configs/bi.yaml")
        >>> validated = validate_config(cfg.train, TrainConfig)
        >>> print(validated.lr)  # Type-safe access
    """
    # Convert OmegaConf to dict if needed
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = cfg

    # Validate and return
    return schema_cls.model_validate(cfg_dict)
