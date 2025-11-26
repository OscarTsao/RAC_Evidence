"""Train 5-fold evidence CE with hybrid negatives, temperature scaling, and OOF predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from Project.evidence import load_evidence_data, train_5fold_evidence
from Project.metrics.eval_evidence import save_evidence_metrics
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger


def main(cfg_path: str, exp: str | None = None) -> None:
    """Run 5-fold evidence training pipeline.

    Args:
        cfg_path: Path to config file
        exp: Experiment name (overrides config)
    """
    logger = get_logger(__name__)

    # Load config
    cfg = load_config(cfg_path)

    if exp:
        cfg.exp = exp

    exp_name = cfg.get("exp", "evidence_5fold")
    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)
    output_dir = Path(f"outputs/runs/{exp_name}")

    logger.info(f"Starting 5-fold evidence training: {exp_name}")
    logger.info(f"Raw data: {raw_dir}")
    logger.info(f"Interim data: {interim_dir}")
    logger.info(f"Output: {output_dir}")

    # Load data and create builder
    k_train = _cfg_get(cfg, "train.top_k_train", 100)
    k_infer = _cfg_get(cfg, "train.top_k_infer", 20)
    neg_per_pos = _cfg_get(cfg, "train.neg_per_pos", 6)
    hard_neg_ratio = _cfg_get(cfg, "train.hard_neg_ratio", 0.8)
    random_neg_ratio = _cfg_get(cfg, "train.random_neg_ratio", 0.1)
    cross_post_ratio = _cfg_get(cfg, "train.cross_post_ratio", 0.1)
    xpost_max_frac = _cfg_get(cfg, "train.xpost_max_frac", 0.2)  # STRICT 20% cap
    epoch_refresh = _cfg_get(cfg, "train.epoch_refresh", False)
    seed = cfg.get("seed", 42)

    logger.info(
        f"Loading data with k_train={k_train}, k_infer={k_infer}, "
        f"neg_per_pos={neg_per_pos}, ratios=({hard_neg_ratio:.0%}/"
        f"{random_neg_ratio:.0%}/{cross_post_ratio:.0%})"
    )

    data_builder = load_evidence_data(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        k_train=k_train,
        neg_per_pos=neg_per_pos,
        hard_neg_ratio=hard_neg_ratio,
        random_neg_ratio=random_neg_ratio,
        cross_post_ratio=cross_post_ratio,
        xpost_max_frac=xpost_max_frac,
        seed=seed,
        epoch_refresh=epoch_refresh,
    )

    # Train 5-fold
    n_folds = _cfg_get(cfg, "train.n_folds", 5)

    result = train_5fold_evidence(
        data_builder=data_builder,
        cfg=cfg,
        output_dir=output_dir,
        n_folds=n_folds,
        seed=seed,
        k_infer=k_infer,
    )

    logger.info(f"Training complete!")
    logger.info(f"OOF predictions: {result['oof_path']}")
    logger.info(f"Mean temperature: {result['mean_temperature']:.4f}")

    # Evaluate OOF predictions
    import json

    with open(result["oof_path"], "r") as f:
        oof_predictions = [json.loads(line) for line in f]

    metrics_path = output_dir / "evidence_metrics.json"
    save_evidence_metrics(oof_predictions, metrics_path)

    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to config file")
    parser.add_argument("--exp", help="Experiment name (overrides config)")
    args = parser.parse_args()

    main(args.cfg, args.exp)
