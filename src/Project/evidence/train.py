"""5-fold evidence CE training with temperature scaling and OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from Project.calib.temperature import TemperatureScaler
from Project.evidence.data_builder import EvidenceDataBuilder, SCExample
from Project.utils.hydra_utils import cfg_get as _cfg_get
from Project.utils.logging import get_logger


def _pair_text(sent_text: str, crit_desc: str) -> str:
    """Pair sentence with criterion using [SEP] token."""
    return f"{sent_text} [SEP] {crit_desc}"


def train_fold(
    train_examples: List[SCExample],
    dev_examples: List[SCExample],
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
    max_length = _cfg_get(cfg, "model.max_length", 384)

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
    dev_dataset = Dataset.from_list(dev_ds).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    # Training arguments
    fold_dir = output_dir / f"fold_{fold}"
    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=_cfg_get(cfg, "train.epochs", 2),
        per_device_train_batch_size=_cfg_get(cfg, "train.batch_size", 8),
        per_device_eval_batch_size=_cfg_get(cfg, "train.batch_size", 8),
        gradient_accumulation_steps=_cfg_get(cfg, "train.gradient_accumulation", 4),
        learning_rate=_cfg_get(cfg, "train.lr", 1.5e-5),
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

    # Fit global temperature (fallback)
    scaler = TemperatureScaler()
    global_temp = scaler.fit(dev_logits, dev_labels)
    logger.info(f"Fold {fold} global temperature: {global_temp:.4f}")

    # Fit per-class temperatures
    per_class_temps = {}
    cid_to_idx = {}
    for i, (logit, label, cid) in enumerate(zip(dev_logits, dev_labels, dev_cids)):
        if cid not in cid_to_idx:
            cid_to_idx[cid] = []
        cid_to_idx[cid].append(i)

    for cid, indices in cid_to_idx.items():
        cid_logits = [dev_logits[i] for i in indices]
        cid_labels = [dev_labels[i] for i in indices]

        if len(cid_logits) < 10:  # Skip small classes
            per_class_temps[cid] = global_temp
            logger.info(f"  {cid}: {len(cid_logits)} samples, using global temp")
        else:
            temp_scaler = TemperatureScaler()
            cid_temp = temp_scaler.fit(cid_logits, cid_labels)
            per_class_temps[cid] = cid_temp
            logger.info(f"  {cid}: {len(cid_logits)} samples, temp={cid_temp:.4f}")

    temperature_dict = {
        "global": global_temp,
        "per_class": per_class_temps
    }

    return ckpt_path, temperature_dict


def generate_oof_predictions(
    test_examples: List[SCExample],
    checkpoint_path: Path,
    temperature_dict: Dict,
    cfg: DictConfig,
) -> List[Dict]:
    """Generate OOF predictions for test fold.

    Args:
        test_examples: Test examples
        checkpoint_path: Path to model checkpoint
        temperature_dict: Dict with 'global' and 'per_class' temperatures
        cfg: Config

    Returns:
        List of predictions with logits and probs
    """
    logger = get_logger(__name__)

    model_name = _cfg_get(cfg, "model.name", "BAAI/bge-reranker-v2-m3")
    max_length = _cfg_get(cfg, "model.max_length", 384)
    batch_size = _cfg_get(cfg, "train.batch_size", 8)

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
                        "sent_id": ex.sent_id,
                        "cid": ex.cid,
                        "label": ex.label,
                        "logit": float(logit),
                        "prob_uncal": p_uncal,
                        "prob_cal": p_cal,
                        "prob": p_cal,  # Legacy support
                        "text": _pair_text(ex.text, ex.criterion),
                    }
                )

    logger.info(f"Generated {len(predictions)} OOF predictions")
    return predictions


def train_5fold_evidence(
    data_builder: EvidenceDataBuilder,
    cfg: DictConfig,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
    k_infer: int = 20,
) -> Dict:
    """Train 5-fold evidence CE with temperature scaling and OOF predictions.

    Args:
        data_builder: Evidence data builder
        cfg: Config
        output_dir: Output directory
        n_folds: Number of folds
        seed: Random seed
        k_infer: Top-K candidates for inference

    Returns:
        Dict with metrics and paths
    """
    logger = get_logger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all post IDs
    post_ids = list(set(lab.post_id for lab in data_builder.labels_sc))
    post_ids.sort()  # Ensure consistent ordering

    # 5-fold split
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_oof_predictions = []
    fold_temperatures = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(post_ids)):
        logger.info(f"=== Fold {fold + 1}/{n_folds} ===")

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

        # Build training / calibration / inference examples
        train_examples = data_builder.build_examples(train_posts)
        dev_examples = data_builder.build_inference_examples(dev_posts, k_infer=k_infer)
        if not dev_examples:
            logger.warning(
                "Fold %d dev posts produced no retrieval candidates; falling back to sampled dev set",
                fold,
            )
            dev_examples = data_builder.build_examples(dev_posts)

        test_examples = data_builder.build_inference_examples(test_posts, k_infer=k_infer)
        if not test_examples:
            raise ValueError(
                f"Fold {fold} has no retrieval candidates for its test posts. "
                "Ensure retrieval_sc.jsonl covers every post."
            )

        # Train fold
        ckpt_path, temperature = train_fold(
            train_examples, dev_examples, cfg, output_dir, fold
        )
        fold_temperatures.append(temperature)

        # Generate OOF predictions
        oof_preds = generate_oof_predictions(test_examples, ckpt_path, temperature, cfg)
        all_oof_predictions.extend(oof_preds)

    # Save OOF predictions
    oof_path = output_dir / "oof_predictions.jsonl"
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
        "oof_path": oof_path,
        "temperatures_path": temps_path,
        "n_predictions": len(all_oof_predictions),
        "mean_temperature": global_mean,
        "k_infer": k_infer,
    }
