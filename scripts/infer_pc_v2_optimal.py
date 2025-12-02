"""Inference for PC v2 with optimal per-class thresholds.

Generates predictions using trained PC v2 models and applies
optimal thresholds learned from OOF predictions for maximum F1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Project.calib.threshold_optimizer import ThresholdOptimizer
from Project.utils.cv_utils import apply_temperature_to_logits
from Project.utils.hydra_utils import cfg_get as _cfg_get
from Project.utils.logging import get_logger


def _pair_text(post_text: str, crit_desc: str) -> str:
    """Pair post with criterion using [SEP] token."""
    return f"{post_text} [SEP] {crit_desc}"


def infer_pc_v2_optimal(
    model_dir: Path,
    posts_path: Path,
    criteria_path: Path,
    thresholds_path: Path,
    temperatures_path: Path,
    output_path: Path,
    cfg: DictConfig,
):
    """Generate PC predictions with optimal thresholds.

    Args:
        model_dir: Directory containing fold_0/best_model, fold_1/best_model, etc.
        posts_path: Path to posts CSV
        criteria_path: Path to criteria JSON
        thresholds_path: Path to optimal_thresholds.json
        temperatures_path: Path to fold_temperatures.json
        output_path: Output path for predictions
        cfg: Configuration
    """
    logger = get_logger(__name__)

    model_name = _cfg_get(cfg, "model.name", "BAAI/bge-reranker-v2-m3")
    max_length = _cfg_get(cfg, "model.max_length", 512)
    batch_size = _cfg_get(cfg, "train.batch_size", 16)
    n_folds = _cfg_get(cfg, "cv.n_folds", 5)

    # Load thresholds
    logger.info(f"Loading optimal thresholds from {thresholds_path}...")
    threshold_optimizer = ThresholdOptimizer.load(thresholds_path)
    thresholds = threshold_optimizer.thresholds_
    logger.info(f"Loaded {len(thresholds)} class-specific thresholds")

    # Load temperatures
    logger.info(f"Loading temperatures from {temperatures_path}...")
    with open(temperatures_path) as f:
        temps_data = json.load(f)

    # Use mean temperatures across folds
    global_temp = temps_data["global"]["mean"]
    per_class_temps = temps_data["per_class_mean"]
    temps = {"global": global_temp, "per_class": per_class_temps}
    logger.info(f"Using mean global temperature: {global_temp:.4f}")

    # Load posts
    logger.info(f"Loading posts from {posts_path}...")
    posts_df = pd.read_csv(posts_path)
    post_lookup = dict(zip(posts_df["post_id"], posts_df["text"]))
    logger.info(f"Loaded {len(post_lookup)} posts")

    # Load criteria
    logger.info(f"Loading criteria from {criteria_path}...")
    with open(criteria_path) as f:
        criteria_data = json.load(f)

    # Map A.1->c1, A.2->c2, ..., A.9->c9 (exclude A.10)
    crit_lookup = {}
    for crit in criteria_data["criteria"]:
        crit_id = crit["id"]
        if crit_id.startswith("A.") and crit_id != "A.10":
            num = crit_id.split(".")[1]
            cid = f"c{num}"
            crit_lookup[cid] = crit["text"]

    logger.info(f"Loaded {len(crit_lookup)} criteria (c1-c9)")

    # Generate predictions for all (post, cid) pairs
    all_predictions = []

    # Create all examples
    examples = []
    for post_id, post_text in post_lookup.items():
        for cid, crit_desc in crit_lookup.items():
            examples.append({
                "post_id": post_id,
                "cid": cid,
                "text": _pair_text(post_text, crit_desc),
            })

    logger.info(f"Created {len(examples)} (post, criterion) pairs")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Generate predictions with each fold model and ensemble
    fold_predictions = {i: [] for i in range(n_folds)}

    for fold in range(n_folds):
        logger.info(f"\nFold {fold}/{n_folds}...")
        model_path = model_dir / f"fold_{fold}" / "best_model"

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}, skipping fold {fold}")
            continue

        logger.info(f"Loading model from {model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        # Generate predictions
        logger.info(f"Generating {len(examples)} predictions...")
        with torch.no_grad():
            for i in range(0, len(examples), batch_size):
                batch_examples = examples[i : i + batch_size]
                batch_texts = [ex["text"] for ex in batch_examples]

                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Forward
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1).cpu().tolist()

                # Store fold predictions
                for ex, logit in zip(batch_examples, logits):
                    fold_predictions[fold].append({
                        "post_id": ex["post_id"],
                        "cid": ex["cid"],
                        "logit": float(logit),
                    })

        logger.info(f"Fold {fold} complete: {len(fold_predictions[fold])} predictions")

    # Ensemble predictions across folds
    logger.info("\nEnsembling predictions across folds...")

    # Group by (post_id, cid)
    from collections import defaultdict
    grouped = defaultdict(list)
    for fold in range(n_folds):
        if not fold_predictions[fold]:
            continue
        for pred in fold_predictions[fold]:
            key = (pred["post_id"], pred["cid"])
            grouped[key].append(pred["logit"])

    # Compute mean logits
    for key, logits in grouped.items():
        post_id, cid = key
        mean_logit = float(sum(logits) / len(logits))

        # Apply temperature scaling
        cid_temp = per_class_temps.get(cid, global_temp)
        safe_temp = max(cid_temp, 1e-4)
        prob_cal = float(torch.sigmoid(torch.tensor(mean_logit) / safe_temp).item())

        # Apply optimal threshold
        threshold = thresholds.get(cid, 0.5)
        pred_label = 1 if prob_cal >= threshold else 0

        all_predictions.append({
            "post_id": post_id,
            "cid": cid,
            "logit": mean_logit,
            "prob_cal": prob_cal,
            "threshold": threshold,
            "prediction": pred_label,
        })

    logger.info(f"\nGenerated {len(all_predictions)} final predictions")

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info(f"Saved predictions to {output_path}")

    # Log summary statistics
    n_positive = sum(1 for p in all_predictions if p["prediction"] == 1)
    n_negative = len(all_predictions) - n_positive
    logger.info(f"\nPrediction Summary:")
    logger.info(f"  Total: {len(all_predictions)}")
    logger.info(f"  Positive: {n_positive} ({n_positive/len(all_predictions)*100:.1f}%)")
    logger.info(f"  Negative: {n_negative} ({n_negative/len(all_predictions)*100:.1f}%)")

    # Per-class summary
    logger.info(f"\nPer-Class Predictions:")
    from collections import Counter
    by_class = defaultdict(list)
    for pred in all_predictions:
        by_class[pred["cid"]].append(pred["prediction"])

    for cid in sorted(by_class.keys()):
        preds = by_class[cid]
        n_pos = sum(preds)
        n_total = len(preds)
        thresh = thresholds.get(cid, 0.5)
        logger.info(f"  {cid}: {n_pos}/{n_total} positive ({n_pos/n_total*100:.1f}%), threshold={thresh:.4f}")

    return all_predictions


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True, help="Model directory")
    parser.add_argument("--posts", type=Path, required=True, help="Posts CSV path")
    parser.add_argument("--criteria", type=Path, required=True, help="Criteria JSON path")
    parser.add_argument("--thresholds", type=Path, required=True, help="Optimal thresholds JSON")
    parser.add_argument("--temperatures", type=Path, required=True, help="Fold temperatures JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions path")
    parser.add_argument("--cfg", type=Path, help="Config file (optional)")

    args = parser.parse_args()

    # Load config if provided, otherwise use defaults
    if args.cfg and args.cfg.exists():
        cfg = OmegaConf.load(args.cfg)
    else:
        cfg = OmegaConf.create({
            "model": {"name": "BAAI/bge-reranker-v2-m3", "max_length": 512},
            "train": {"batch_size": 16},
            "cv": {"n_folds": 5},
        })

    infer_pc_v2_optimal(
        model_dir=args.model_dir,
        posts_path=args.posts,
        criteria_path=args.criteria,
        thresholds_path=args.thresholds,
        temperatures_path=args.temperatures,
        output_path=args.output,
        cfg=cfg,
    )
