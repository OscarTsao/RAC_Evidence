"""Prepare data splits and sanity checks."""

from __future__ import annotations

import json
from pathlib import Path

from Project.dataio.splits import split_posts
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.logging import get_logger


def prepare_data_splits(cfg_path: str) -> None:
    logger = get_logger(__name__)
    cfg = load_config(cfg_path)
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    dataset = load_raw_dataset(raw_dir)
    splits = split_posts(dataset["posts"], cfg.split.ratios, seed=cfg.split.seed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    logger.info("Wrote splits to %s", processed_dir / "splits.json")


