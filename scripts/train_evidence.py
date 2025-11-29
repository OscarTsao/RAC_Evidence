"""HPO for Evidence Cross-Encoder with Optuna."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner

from Project.evidence import load_evidence_data, train_5fold_evidence
from Project.metrics.eval_evidence import save_evidence_metrics
from Project.utils.hydra_utils import load_config
from Project.utils.logging import get_logger

logger = get_logger(__name__)


def objective(trial: optuna.Trial, base_cfg: dict, exp: str) -> float:
    """Objective function for HPO.

    Returns macro_f1 (to maximize), but prunes if precision@5 < 0.80

    Args:
        trial: Optuna trial object
        base_cfg: Base configuration dictionary
        exp: Experiment name

    Returns:
        Macro-F1 score (primary objective)
    """
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 7e-6, 2.5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    epochs = trial.suggest_int("epochs", 2, 4)

    # LoRA
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.15)

    # Data
    neg_per_pos = trial.suggest_categorical("neg_per_pos", [6, 8])
    xpost_neg_frac = trial.suggest_float("xpost_neg_frac", 0.0, 0.25)
    context_mode = trial.suggest_categorical("context_mode", ["none", "neighbors1"])
    sent_max_len = trial.suggest_categorical("sent_max_len", [256, 384])

    # Loss (conditional)
    loss_type = trial.suggest_categorical("loss_type", ["focal", "bce_ranking"])

    if loss_type == "focal":
        focal_gamma = trial.suggest_float("focal_gamma", 1.5, 2.5)
        focal_alpha = trial.suggest_float("focal_alpha", 0.6, 0.85)
        rank_weight = 0.5
        rank_margin = 0.3
        pos_weight_scale = 1.0
    else:  # bce_ranking
        rank_weight = trial.suggest_float("rank_weight", 0.2, 0.6)
        rank_margin = trial.suggest_float("rank_margin", 0.1, 0.4)
        pos_weight_scale = trial.suggest_float("pos_weight_scale", 1.0, 6.0)
        focal_gamma = 2.0
        focal_alpha = 0.75

    # Build config for this trial
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(base_cfg)

    # Update config with trial hyperparameters
    cfg.train.lr = lr
    cfg.train.batch_size = batch_size
    cfg.train.epochs = epochs
    cfg.model.lora.r = lora_r
    cfg.model.lora.alpha = lora_alpha
    cfg.model.lora.dropout = lora_dropout
    cfg.train.neg_per_pos = neg_per_pos
    cfg.train.cross_post_ratio = xpost_neg_frac
    cfg.data.context_mode = context_mode
    cfg.model.max_length = sent_max_len
    cfg.train.loss_type = loss_type
    cfg.train.focal_gamma = focal_gamma
    cfg.train.focal_alpha = focal_alpha
    cfg.train.rank_weight = rank_weight
    cfg.train.rank_margin = rank_margin
    cfg.train.pos_weight_scale = pos_weight_scale

    # Run training
    output_dir = Path(f"outputs/runs/{exp}_trial_{trial.number}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        data_builder = load_evidence_data(
            raw_dir=Path(cfg.data.raw_dir),
            interim_dir=Path(cfg.data.interim_dir),
            k_train=cfg.train.top_k_train,
            neg_per_pos=cfg.train.neg_per_pos,
            hard_neg_ratio=cfg.train.hard_neg_ratio,
            random_neg_ratio=cfg.train.random_neg_ratio,
            cross_post_ratio=cfg.train.cross_post_ratio,
            xpost_max_frac=cfg.train.xpost_max_frac,
            seed=cfg.seed,
            context_mode=cfg.data.get("context_mode", "none"),
        )

        # Train 5-fold
        result = train_5fold_evidence(
            data_builder=data_builder,
            cfg=cfg,
            output_dir=output_dir,
            n_folds=cfg.train.n_folds,
            seed=cfg.seed,
            k_infer=cfg.train.top_k_infer,
        )

        # Evaluate
        with open(result["oof_path"], "r") as f:
            oof_predictions = [json.loads(line) for line in f]

        metrics_path = output_dir / "evidence_metrics.json"
        save_evidence_metrics(oof_predictions, metrics_path)

        with open(metrics_path) as f:
            metrics = json.load(f)["overall"]

        # Check precision constraint
        precision_at_5 = metrics.get("precision@5", 0.0)
        macro_f1 = metrics.get("macro_f1", 0.0)

        # Pruning: fail if precision too low
        if precision_at_5 < 0.80:
            logger.warning(f"Trial {trial.number} pruned: precision@5={precision_at_5:.4f} < 0.80")
            return 0.0

        # Log metrics to trial
        trial.set_user_attr("precision@5", precision_at_5)
        trial.set_user_attr("macro_f1", macro_f1)
        trial.set_user_attr("f1", metrics.get("f1", 0.0))
        trial.set_user_attr("ece", metrics.get("ece", 1.0))

        logger.info(
            f"Trial {trial.number}: precision@5={precision_at_5:.4f}, "
            f"macro_f1={macro_f1:.4f}, f1={metrics.get('f1', 0):.4f}"
        )

        return macro_f1

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0.0


def main(cfg_path: str, exp: str, n_trials: int = 30) -> None:
    """Run HPO with Optuna.

    Args:
        cfg_path: Path to base configuration file
        exp: Experiment name
        n_trials: Number of trials to run
    """
    logger.info(f"Starting HPO for {exp} with {n_trials} trials")

    # Load base config
    cfg = load_config(cfg_path)
    cfg_dict = dict(cfg)

    # Create study
    study = optuna.create_study(
        study_name=f"{exp}_hpo",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, cfg_dict, exp),
        n_trials=n_trials,
        n_jobs=1,  # Sequential for reproducibility
    )

    # Report best trial
    logger.info("=" * 80)
    logger.info("HPO Complete!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best macro_f1: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Report user attributes for best trial
    logger.info("Best trial metrics:")
    for key, value in study.best_trial.user_attrs.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("=" * 80)

    # Save best config
    best_cfg_path = Path(f"configs/{exp}_best_hpo.yaml")
    logger.info(f"Best config can be saved to {best_cfg_path}")

    # Save hyperparameters to JSON for reference
    best_params_path = Path(f"outputs/{exp}_best_params.json")
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "trial_number": study.best_trial.number,
                "macro_f1": study.best_value,
                "params": study.best_params,
                "user_attrs": study.best_trial.user_attrs,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved best hyperparameters to {best_params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Base config path")
    parser.add_argument("--exp", required=True, help="Experiment name")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of HPO trials")
    args = parser.parse_args()

    main(args.cfg, args.exp, args.n_trials)
