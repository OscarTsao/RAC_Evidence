"""Train the bi-encoder model."""

from __future__ import annotations

from pathlib import Path

from Project.retrieval.train_bi import train_bi_encoder
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger


def main(cfg_path: str) -> None:
    logger = get_logger(__name__)
    cfg = load_config(cfg_path)
    raw_dir = Path(cfg.data.raw_dir)
    exp = cfg.get("exp", "debug")
    out_dir = Path(cfg.get("output", f"outputs/runs/{exp}")) / "bi"
    dataset = load_raw_dataset(raw_dir)
    train_bi_encoder(
        dataset["sentences"],
        dataset["criteria"],
        out_dir,
        seed=cfg.split.seed,
        model_name=_cfg_get(cfg, "model.name"),
    )
    logger.info("Bi-encoder trained")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
