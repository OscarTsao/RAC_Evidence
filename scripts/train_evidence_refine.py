"""Phase 3 Refined HPO for Evidence Cross-Encoder with Optuna.

This script runs a refined HPO with a tighter search space focused on
fine-tuning the best solution from Phase 2 (Trial #0).

Phase 2 Best (Trial #0):
- Precision@5: 0.80
- F1: 0.5030
- Macro-F1: 0.7475
- ECE: 0.0404
- Loss: bce_ranking with rank_weight=0.488, rank_margin=0.391, pos_weight_scale=1.352

Refined Search Space (5 dimensions):
1. Learning Rate: [1.5e-5, 3.5e-5] (centered on 2.395e-05)
2. Ranking Weight: [0.40, 0.70] (centered on 0.488)
3. Ranking Margin: [0.30, 0.50] (centered on 0.391)
4. LoRA Dropout: [0.05, 0.15] (centered on 0.086)
5. Data Aggression: neg_per_pos ∈ {8, 10}

Fixed Parameters:
- loss_type: "bce_ranking" (no focal loss)
- batch_size: 16
- epochs: 3
- lora_r: 16
- lora_alpha: 16
- context_mode: "none"
- sent_max_len: 256
- xpost_neg_frac: 0.177
- pos_weight_scale: 1.352

Hard Constraint:
- Prune if Precision@5 < 0.80

Objective:
- Primary: Maximize Macro-F1
"""

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
    """Objective function for refined HPO.

    Fixed parameters from Phase 2 Trial #0, with refined search on:
    - lr, rank_weight, rank_margin, lora_dropout, neg_per_pos

    Returns macro_f1 (to maximize), but prunes if precision@5 < 0.80

    Args:
        trial: Optuna trial object
        base_cfg: Base configuration dictionary
        exp: Experiment name

    Returns:
        Macro-F1 score (primary objective)
    """
    # Refined search space (5 dimensions)
    lr = trial.suggest_float("lr", 1.5e-5, 3.5e-5, log=True)
    rank_weight = trial.suggest_float("rank_weight", 0.40, 0.70)
    rank_margin = trial.suggest_float("rank_margin", 0.30, 0.50)
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.15)
    neg_per_pos = trial.suggest_categorical("neg_per_pos", [8, 10])

    # Fixed parameters from Phase 2 Trial #0
    batch_size = 16
    epochs = 3
    lora_r = 16
    lora_alpha = 16
    context_mode = "none"
    sent_max_len = 256
    xpost_neg_frac = 0.177
    loss_type = "bce_ranking"
    pos_weight_scale = 1.352
    focal_gamma = 2.0  # Not used but keep for compatibility
    focal_alpha = 0.75  # Not used but keep for compatibility

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
        f1 = metrics.get("f1", 0.0)
        ece = metrics.get("ece", 1.0)

        # Pruning: fail if precision too low
        if precision_at_5 < 0.80:
            logger.warning(f"Trial {trial.number} pruned: precision@5={precision_at_5:.4f} < 0.80")
            return 0.0

        # Log metrics to trial
        trial.set_user_attr("precision@5", precision_at_5)
        trial.set_user_attr("macro_f1", macro_f1)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("ece", ece)

        logger.info(
            f"Trial {trial.number}: precision@5={precision_at_5:.4f}, "
            f"macro_f1={macro_f1:.4f}, f1={f1:.4f}, ece={ece:.4f}"
        )
        logger.info(
            f"  lr={lr:.2e}, rank_weight={rank_weight:.3f}, "
            f"rank_margin={rank_margin:.3f}, lora_dropout={lora_dropout:.3f}, neg_per_pos={neg_per_pos}"
        )

        return macro_f1

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0.0


def main(cfg_path: str, exp: str, n_trials: int = 15) -> None:
    """Run refined HPO with Optuna.

    Args:
        cfg_path: Path to base configuration file
        exp: Experiment name
        n_trials: Number of trials to run (default: 15)
    """
    logger.info("=" * 80)
    logger.info("Phase 3 Refined HPO for Evidence Cross-Encoder")
    logger.info("=" * 80)
    logger.info(f"Experiment: {exp}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info("")
    logger.info("Refined Search Space (5 dimensions):")
    logger.info("  1. Learning Rate: [1.5e-5, 3.5e-5] (log scale)")
    logger.info("  2. Ranking Weight: [0.40, 0.70]")
    logger.info("  3. Ranking Margin: [0.30, 0.50]")
    logger.info("  4. LoRA Dropout: [0.05, 0.15]")
    logger.info("  5. Data Aggression: neg_per_pos ∈ {8, 10}")
    logger.info("")
    logger.info("Fixed Parameters (from Phase 2 Trial #0):")
    logger.info("  - loss_type: bce_ranking")
    logger.info("  - batch_size: 16")
    logger.info("  - epochs: 3")
    logger.info("  - lora_r: 16")
    logger.info("  - lora_alpha: 16")
    logger.info("  - context_mode: none")
    logger.info("  - sent_max_len: 256")
    logger.info("  - xpost_neg_frac: 0.177")
    logger.info("  - pos_weight_scale: 1.352")
    logger.info("")
    logger.info("Objective: Maximize Macro-F1")
    logger.info("Hard Constraint: Precision@5 >= 0.80")
    logger.info("=" * 80)
    logger.info("")

    # Load base config
    cfg = load_config(cfg_path)
    cfg_dict = dict(cfg)

    # Create study
    study = optuna.create_study(
        study_name=f"{exp}_hpo",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=0),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, cfg_dict, exp),
        n_trials=n_trials,
        n_jobs=1,  # Sequential for reproducibility
    )

    # Report best trial
    logger.info("")
    logger.info("=" * 80)
    logger.info("HPO Complete!")
    logger.info("=" * 80)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best macro_f1: {study.best_value:.4f}")
    logger.info("")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4e}" if value < 0.01 else f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Report user attributes for best trial
    logger.info("")
    logger.info("Best trial metrics:")
    for key, value in study.best_trial.user_attrs.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Comparison with Phase 2
    logger.info("")
    logger.info("=" * 80)
    logger.info("Comparison with Phase 2 Trial #0:")
    logger.info("=" * 80)
    logger.info("  Phase 2 Trial #0:")
    logger.info("    Precision@5: 0.8000")
    logger.info("    F1: 0.5030")
    logger.info("    Macro-F1: 0.7475")
    logger.info("    ECE: 0.0404")
    logger.info("")
    logger.info("  Phase 3 Refined Best:")
    logger.info(f"    Precision@5: {study.best_trial.user_attrs['precision@5']:.4f}")
    logger.info(f"    F1: {study.best_trial.user_attrs['f1']:.4f}")
    logger.info(f"    Macro-F1: {study.best_value:.4f}")
    logger.info(f"    ECE: {study.best_trial.user_attrs['ece']:.4f}")
    logger.info("")

    # Determine conclusion
    phase2_f1 = 0.5030
    phase2_macro_f1 = 0.7475
    refined_f1 = study.best_trial.user_attrs["f1"]
    refined_macro_f1 = study.best_value
    refined_precision = study.best_trial.user_attrs["precision@5"]

    if refined_precision >= 0.80:
        if abs(refined_f1 - phase2_f1) <= 0.01:
            conclusion = "STABLE - Solution validated as robust"
        elif refined_f1 > phase2_f1 + 0.01:
            conclusion = "IMPROVED - Better solution found"
        else:
            conclusion = "REGRESSION - Performance degraded slightly"
    else:
        conclusion = "FAILED - Precision@5 constraint not met"

    logger.info(f"Conclusion: {conclusion}")
    logger.info("=" * 80)
    logger.info("")

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
                "phase2_comparison": {
                    "phase2_f1": phase2_f1,
                    "phase2_macro_f1": phase2_macro_f1,
                    "phase2_precision@5": 0.80,
                    "phase2_ece": 0.0404,
                    "conclusion": conclusion,
                },
            },
            f,
            indent=2,
        )
    logger.info(f"Saved best hyperparameters to {best_params_path}")
    logger.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Base config path")
    parser.add_argument("--exp", required=True, help="Experiment name")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of HPO trials")
    args = parser.parse_args()

    main(args.cfg, args.exp, args.n_trials)
