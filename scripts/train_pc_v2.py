"""5-fold PC (Post-Criterion) CE training with temperature scaling and OOF predictions.

This script mirrors the successful Evidence pipeline architecture with:
- Proper 5-fold CV with true OOF predictions
- Conservative hyperparameters to prevent model collapse
- Per-class temperature scaling on dev set
- Full post context (max_length=512)

Key differences from broken train_ce_pc.py:
- Split data BEFORE fold loop (not inside)
- Conservative LR (2e-5 not 1e-3)
- Proper dev/test separation for calibration
- No model collapse (diverse predictions)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from Project.utils.cv_utils import fit_per_class_temperature
from Project.utils.hydra_utils import cfg_get as _cfg_get
from Project.utils.logging import get_logger


@dataclass
class PCExample:
    """Post-criterion training example."""

    post_id: str
    cid: str  # c1-c9
    label: int
    text: str  # full post text
    criterion: str  # criterion description


class PCDataBuilder:
    """Build PC training data from labels, posts, and criteria."""

    def __init__(
        self,
        labels_path: Path,
        posts_path: Path,
        criteria_path: Path,
    ):
        """Initialize PC data builder.

        Args:
            labels_path: Path to labels_pc.jsonl
            posts_path: Path to redsm5_posts.csv
            criteria_path: Path to MDD_Criteira.json
        """
        self.logger = get_logger(__name__)

        # Load labels
        self.labels = []
        with open(labels_path, "r") as f:
            for line in f:
                data = json.loads(line)
                self.labels.append(
                    {
                        "post_id": data["post_id"],
                        "cid": data["cid"],
                        "label": int(data["label"]),
                    }
                )
        self.logger.info(f"Loaded {len(self.labels)} PC labels from {labels_path}")

        # Load posts
        posts_df = pd.read_csv(posts_path)
        self.post_lookup = dict(zip(posts_df["post_id"], posts_df["text"]))
        self.logger.info(f"Loaded {len(self.post_lookup)} posts from {posts_path}")

        # Load criteria and build A.X -> cX mapping
        with open(criteria_path, "r") as f:
            criteria_data = json.load(f)

        # Map A.1->c1, A.2->c2, ..., A.9->c9 (exclude A.10)
        self.crit_lookup = {}
        for crit in criteria_data["criteria"]:
            crit_id = crit["id"]
            if crit_id.startswith("A.") and crit_id != "A.10":
                # Extract number: A.1 -> 1
                num = crit_id.split(".")[1]
                cid = f"c{num}"
                self.crit_lookup[cid] = crit["text"]

        self.logger.info(
            f"Loaded {len(self.crit_lookup)} criteria: {sorted(self.crit_lookup.keys())}"
        )

    def build_examples(self, post_ids: List[str]) -> List[PCExample]:
        """Build PC examples for given post IDs.

        Args:
            post_ids: List of post IDs to include

        Returns:
            List of PC examples
        """
        post_ids_set = set(post_ids)
        examples = []

        # Filter labels for specified posts
        for label_rec in self.labels:
            if label_rec["post_id"] not in post_ids_set:
                continue

            post_id = label_rec["post_id"]
            cid = label_rec["cid"]
            label = label_rec["label"]

            # Get post text and criterion description
            post_text = self.post_lookup.get(post_id)
            crit_desc = self.crit_lookup.get(cid)

            if post_text is None:
                self.logger.warning(f"Missing post text for {post_id}")
                continue

            if crit_desc is None:
                self.logger.warning(f"Missing criterion for {cid}")
                continue

            examples.append(
                PCExample(
                    post_id=post_id,
                    cid=cid,
                    label=label,
                    text=post_text,
                    criterion=crit_desc,
                )
            )

        self.logger.info(f"Built {len(examples)} PC examples from {len(post_ids)} posts")

        # Log class distribution
        n_pos = sum(1 for ex in examples if ex.label == 1)
        n_neg = len(examples) - n_pos
        self.logger.info(f"Class distribution: {n_pos} positive, {n_neg} negative")

        return examples


def _pair_text(post_text: str, crit_desc: str) -> str:
    """Pair post with criterion using [SEP] token."""
    return f"{post_text} [SEP] {crit_desc}"


def train_fold(
    train_examples: List[PCExample],
    dev_examples: List[PCExample],
    cfg: DictConfig,
    output_dir: Path,
    fold: int,
) -> Tuple[Path, Dict]:
    """Train cross-encoder for one fold.

    Args:
        train_examples: Training examples
        dev_examples: Dev examples (for temperature scaling)
        cfg: Config
        output_dir: Output directory
        fold: Fold number

    Returns:
        Tuple of (checkpoint_path, temperature_dict)
            temperature_dict contains 'global' and 'per_class' temperatures
    """
    logger = get_logger(__name__)

    model_name = _cfg_get(cfg, "model.name", "BAAI/bge-reranker-v2-m3")
    max_length = _cfg_get(cfg, "model.max_length", 512)

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Prepare datasets
    train_texts = [_pair_text(ex.text, ex.criterion) for ex in train_examples]
    train_labels = [ex.label for ex in train_examples]

    dev_texts = [_pair_text(ex.text, ex.criterion) for ex in dev_examples]
    dev_labels = [ex.label for ex in dev_examples]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )

    # Create datasets
    train_ds = [{"text": t, "label": float(l)} for t, l in zip(train_texts, train_labels)]
    dev_ds = [{"text": t, "label": float(l)} for t, l in zip(dev_texts, dev_labels)]

    # Tokenize
    from datasets import Dataset

    train_dataset = Dataset.from_list(train_ds).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    dev_dataset = Dataset.from_list(dev_ds).map(tokenize, batched=True, remove_columns=["text"])

    # Training arguments
    fold_dir = output_dir / f"fold_{fold}"
    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=_cfg_get(cfg, "train.epochs", 3),
        per_device_train_batch_size=_cfg_get(cfg, "train.batch_size", 16),
        per_device_eval_batch_size=_cfg_get(cfg, "train.batch_size", 16),
        gradient_accumulation_steps=_cfg_get(cfg, "train.gradient_accumulation", 4),
        learning_rate=_cfg_get(cfg, "train.lr", 2e-5),  # Conservative!
        weight_decay=_cfg_get(cfg, "train.weight_decay", 0.01),
        warmup_ratio=_cfg_get(cfg, "train.warmup_ratio", 0.06),
        logging_steps=_cfg_get(cfg, "train.logging_steps", 50),
        eval_strategy="steps",
        eval_steps=_cfg_get(cfg, "train.eval_steps", 500),
        save_strategy="steps",
        save_steps=_cfg_get(cfg, "train.save_steps", 500),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=_cfg_get(cfg, "train.bf16", True),
        fp16=_cfg_get(cfg, "train.fp16", False),
        gradient_checkpointing=False,
        dataloader_num_workers=_cfg_get(cfg, "dataloader.num_workers", 4),
        dataloader_pin_memory=_cfg_get(cfg, "dataloader.pin_memory", True),
        remove_unused_columns=False,
        report_to="none",  # Disable MLflow/wandb integration
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info(f"Training fold {fold} with {len(train_examples)} examples")
    trainer.train()

    # Save best checkpoint
    ckpt_path = fold_dir / "best_model"
    trainer.save_model(str(ckpt_path))
    logger.info(f"Saved fold {fold} checkpoint to {ckpt_path}")

    # Temperature scaling on dev set
    logger.info(f"Fitting temperature on {len(dev_examples)} dev examples")
    dev_logits = []
    dev_cids = []  # Track criterion IDs
    model.eval()

    with torch.no_grad():
        for i in range(0, len(dev_dataset), training_args.per_device_eval_batch_size):
            batch_data = dev_dataset[i : i + training_args.per_device_eval_batch_size]
            batch_examples = dev_examples[i : i + training_args.per_device_eval_batch_size]
            inputs = data_collator(batch_data)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            dev_logits.extend(logits.cpu().tolist())
            dev_cids.extend([ex.cid for ex in batch_examples])

    temperature_dict = fit_per_class_temperature(dev_logits, dev_labels, dev_cids)
    logger.info(
        "Fold %d temperatures: global=%.4f, per_class=%s",
        fold,
        temperature_dict.get("global", 1.0),
        {k: f"{v:.4f}" for k, v in temperature_dict.get("per_class", {}).items()},
    )

    return ckpt_path, temperature_dict


def generate_oof_predictions(
    test_examples: List[PCExample],
    checkpoint_path: Path,
    temperature_dict: Dict,
    cfg: DictConfig,
    fold: int,
) -> List[Dict]:
    """Generate OOF predictions for test fold.

    Args:
        test_examples: Test examples
        checkpoint_path: Path to model checkpoint
        temperature_dict: Dict with 'global' and 'per_class' temperatures
        cfg: Config
        fold: Fold number

    Returns:
        List of predictions with logits and probs
    """
    logger = get_logger(__name__)

    model_name = _cfg_get(cfg, "model.name", "BAAI/bge-reranker-v2-m3")
    max_length = _cfg_get(cfg, "model.max_length", 512)
    batch_size = _cfg_get(cfg, "train.batch_size", 16)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    predictions = []

    # Generate predictions
    test_texts = [_pair_text(ex.text, ex.criterion) for ex in test_examples]

    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i : i + batch_size]
            batch_examples = test_examples[i : i + batch_size]

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

            # Apply temperature scaling
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            probs_uncal = torch.sigmoid(logits_tensor).tolist()

            # Apply per-class temperature (with global fallback)
            global_temp = temperature_dict.get("global", 1.0)
            per_class_temps = temperature_dict.get("per_class", {})

            probs_cal = []
            for logit_val, ex in zip(logits, batch_examples):
                cid_temp = per_class_temps.get(ex.cid, global_temp)
                safe_temp = max(cid_temp, 1e-4)
                prob = torch.sigmoid(torch.tensor(logit_val) / safe_temp).item()
                probs_cal.append(prob)

            # Store predictions
            for ex, logit, p_uncal, p_cal in zip(batch_examples, logits, probs_uncal, probs_cal):
                predictions.append(
                    {
                        "post_id": ex.post_id,
                        "cid": ex.cid,
                        "label": ex.label,
                        "logit": float(logit),
                        "prob_uncal": p_uncal,
                        "prob_cal": p_cal,
                        "prob": p_cal,  # Legacy support
                        "fold": fold,
                    }
                )

    logger.info(f"Generated {len(predictions)} OOF predictions for fold {fold}")
    return predictions


def train_5fold_pc(
    data_builder: PCDataBuilder,
    cfg: DictConfig,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """Train 5-fold PC CE with temperature scaling and OOF predictions.

    Args:
        data_builder: PC data builder
        cfg: Config
        output_dir: Output directory
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Dict with metrics and paths
    """
    logger = get_logger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all (post_id, cid) pairs with labels for stratification
    # This ensures balanced label distribution across folds
    pairs_and_labels = []
    for label_rec in data_builder.labels:
        post_id = label_rec["post_id"]
        cid = label_rec["cid"]
        label = label_rec["label"]
        pairs_and_labels.append(((post_id, cid), label))

    # Sort for consistent ordering
    pairs_and_labels.sort(key=lambda x: (x[0][0], x[0][1]))

    # Extract unique post IDs
    post_ids = sorted(list(set(p[0] for p, _ in pairs_and_labels)))
    logger.info(f"Total posts: {len(post_ids)}")

    # Create stratification labels: for each post, concatenate all its criterion labels
    # This ensures posts with similar symptom patterns are distributed evenly
    post_to_label_vec = {}
    for post_id in post_ids:
        # Get all labels for this post
        post_labels = [
            label for (pid, _cid), label in pairs_and_labels if pid == post_id
        ]
        # Create a hash of the label pattern (simple: join as string)
        post_to_label_vec[post_id] = "".join(map(str, sorted(post_labels)))

    # Convert to stratification array
    strat_labels = [post_to_label_vec[pid] for pid in post_ids]

    # 5-fold split with stratification
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_oof_predictions = []
    fold_temperatures = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(post_ids, strat_labels)):
        logger.info(f"=== Fold {fold}/{n_folds} ===")

        # Split train into train + dev for temperature scaling
        train_posts = [post_ids[i] for i in train_idx]
        test_posts = [post_ids[i] for i in test_idx]

        # Further split train into train + dev (90/10)
        n_dev = max(1, len(train_posts) // 10)
        dev_posts = train_posts[-n_dev:]
        train_posts = train_posts[:-n_dev]

        logger.info(
            f"Fold {fold}: {len(train_posts)} train, {len(dev_posts)} dev, {len(test_posts)} test posts"
        )

        # Build examples
        train_examples = data_builder.build_examples(train_posts)
        dev_examples = data_builder.build_examples(dev_posts)
        test_examples = data_builder.build_examples(test_posts)

        # Train fold
        ckpt_path, temperature = train_fold(train_examples, dev_examples, cfg, output_dir, fold)
        fold_temperatures.append(temperature)

        # Generate OOF predictions
        oof_preds = generate_oof_predictions(test_examples, ckpt_path, temperature, cfg, fold)
        all_oof_predictions.extend(oof_preds)

    # Save OOF predictions
    oof_path = output_dir / "pc_oof.jsonl"
    with open(oof_path, "w") as f:
        for pred in all_oof_predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info(f"Saved {len(all_oof_predictions)} OOF predictions to {oof_path}")

    # Save fold temperatures
    temps_path = output_dir / "fold_temperatures.json"
    global_temps = []
    per_class_temps: Dict[str, List[float]] = {}
    for temp in fold_temperatures:
        if isinstance(temp, dict):
            global_temps.append(float(temp.get("global", 1.0)))
            for cid, cid_temp in temp.get("per_class", {}).items():
                per_class_temps.setdefault(cid, []).append(float(cid_temp))
        else:
            global_temps.append(float(temp))

    global_mean = float(np.mean(global_temps)) if global_temps else 1.0
    global_std = float(np.std(global_temps)) if global_temps else 0.0
    per_class_mean = {cid: float(np.mean(vals)) for cid, vals in per_class_temps.items()}
    per_class_std = {cid: float(np.std(vals)) for cid, vals in per_class_temps.items()}

    with open(temps_path, "w") as f:
        json.dump(
            {
                "temperatures": fold_temperatures,
                "global": {
                    "per_fold": global_temps,
                    "mean": global_mean,
                    "std": global_std,
                },
                "per_class_mean": per_class_mean,
                "per_class_std": per_class_std,
            },
            f,
            indent=2,
        )

    logger.info("Fold global temperatures: %s", global_temps)
    logger.info("Mean global temperature: %.4f", global_mean)

    return {
        "oof_path": str(oof_path),
        "temperatures_path": str(temps_path),
        "n_predictions": len(all_oof_predictions),
        "mean_temperature": global_mean,
    }


@hydra.main(config_path="../configs", config_name="pc_v2", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    logger = get_logger(__name__)
    logger.info("Starting PC v2 training pipeline")

    # Parse config
    labels_path = Path(_cfg_get(cfg, "data.labels_path"))
    posts_path = Path(_cfg_get(cfg, "data.posts_path"))
    criteria_path = Path(_cfg_get(cfg, "data.criteria_path"))
    output_dir = Path(_cfg_get(cfg, "output_dir"))
    seed = _cfg_get(cfg, "seed", 42)
    n_folds = _cfg_get(cfg, "cv.n_folds", 5)

    # Verify files exist
    assert labels_path.exists(), f"Labels file not found: {labels_path}"
    assert posts_path.exists(), f"Posts file not found: {posts_path}"
    assert criteria_path.exists(), f"Criteria file not found: {criteria_path}"

    # Create data builder
    data_builder = PCDataBuilder(
        labels_path=labels_path,
        posts_path=posts_path,
        criteria_path=criteria_path,
    )

    # Train 5-fold
    results = train_5fold_pc(
        data_builder=data_builder,
        cfg=cfg,
        output_dir=output_dir,
        n_folds=n_folds,
        seed=seed,
    )

    logger.info("Training complete!")
    logger.info(f"OOF predictions saved to: {results['oof_path']}")
    logger.info(f"Fold temperatures saved to: {results['temperatures_path']}")
    logger.info(f"Total predictions: {results['n_predictions']}")
    logger.info(f"Mean global temperature: {results['mean_temperature']:.4f}")


if __name__ == "__main__":
    main()
