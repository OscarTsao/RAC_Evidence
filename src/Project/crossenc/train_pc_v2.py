"""Cross-encoder PC training v2 with strict CV, focal loss, and calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from Project.utils.cv_utils import (
    apply_temperature_to_logits,
    compute_optimal_thresholds,
    fit_per_class_temperature,
    save_json,
)
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get
from Project.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PCExample:
    text: str
    label: int
    post_id: str
    cid: str


def build_pc_examples(raw_dir: Path) -> List[PCExample]:
    """Create PC examples paired with criterion descriptions."""
    data = load_raw_dataset(raw_dir)
    crit_desc = {c.cid: c.desc for c in data["criteria"]}
    post_text = {p.post_id: p.text for p in data["posts"]}
    examples: List[PCExample] = []
    for lab in data["labels_pc"]:
        text = f"{post_text.get(lab.post_id, '')} [SEP] {crit_desc.get(lab.cid, '')}"
        examples.append(
            PCExample(text=text, label=int(lab.label), post_id=lab.post_id, cid=lab.cid)
        )
    return examples


def binary_focal_loss(logits: torch.Tensor, labels: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """Binary focal loss on logits."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return loss.mean()


class FocalTrainer(Trainer):
    """Trainer with binary focal loss."""

    def __init__(self, focal_alpha: float, focal_gamma: float, *args, **kwargs):
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        loss = binary_focal_loss(logits, labels, self.focal_alpha, self.focal_gamma)
        return (loss, outputs) if return_outputs else loss


def tokenize_dataset(ds: List[PCExample], tokenizer, max_length: int) -> Dataset:
    hf_ds = Dataset.from_list(
        [{"text": ex.text, "labels": float(ex.label)} for ex in ds]
    )

    def _tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return enc

    return hf_ds.map(_tok, batched=True, remove_columns=["text"])


def compute_macro_f1(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits.reshape(-1)))
    preds = (probs >= 0.5).astype(int)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def train_fold(
    fold: int,
    train_examples: List[PCExample],
    eval_examples: List[PCExample],
    cfg: DictConfig,
    out_dir: Path,
) -> Tuple[Trainer, Dataset]:
    model_name = _cfg_get(cfg, "model.name", "BAAI/bge-reranker-v2-m3")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    if torch.cuda.is_available():
        model = model.cuda()
        torch.set_float32_matmul_precision("high")

    max_len = int(_cfg_get(cfg, "model.max_length", 512))
    train_ds = tokenize_dataset(train_examples, tokenizer, max_len)
    eval_ds = tokenize_dataset(eval_examples, tokenizer, max_len)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    training_args = TrainingArguments(
        output_dir=str(out_dir / f"fold_{fold}"),
        learning_rate=float(_cfg_get(cfg, "train.lr", 2e-5)),
        per_device_train_batch_size=int(_cfg_get(cfg, "train.batch_size", 16)),
        per_device_eval_batch_size=int(_cfg_get(cfg, "train.batch_size", 16)),
        gradient_accumulation_steps=int(_cfg_get(cfg, "train.gradient_accumulation", 1)),
        num_train_epochs=float(_cfg_get(cfg, "train.epochs", 4)),
        warmup_ratio=float(_cfg_get(cfg, "train.warmup_ratio", 0.06)),
        logging_steps=int(_cfg_get(cfg, "train.logging_steps", 50)),
        eval_strategy="steps",
        eval_steps=int(_cfg_get(cfg, "train.eval_steps", 200)),
        save_strategy="steps",
        save_steps=int(_cfg_get(cfg, "train.save_steps", 200)),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=bool(_cfg_get(cfg, "train.bf16", True)),
        fp16=bool(_cfg_get(cfg, "train.fp16", False)),
        gradient_checkpointing=bool(_cfg_get(cfg, "train.gradient_checkpointing", False)),
        dataloader_num_workers=int(_cfg_get(cfg, "train.num_workers", 4)),
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        report_to=[],
        max_grad_norm=float(_cfg_get(cfg, "train.grad_clip", 1.0)),
    )

    trainer = FocalTrainer(
        focal_alpha=float(_cfg_get(cfg, "train.focal_alpha", 0.9)),
        focal_gamma=float(_cfg_get(cfg, "train.focal_gamma", 2.0)),
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_macro_f1,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(_cfg_get(cfg, "train.early_stopping_patience", 3)))],
    )

    logger.info(
        "Fold %d: training %d examples, eval %d examples", fold, len(train_ds), len(eval_ds)
    )
    trainer.train()
    return trainer, eval_ds


def predict_oof(trainer: Trainer, test_ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    preds = trainer.predict(test_ds)
    logits = preds.predictions.reshape(-1)
    labels = np.array(test_ds["labels"])
    return logits, labels


def compute_metrics_from_oof(oof_rows: List[Dict], thresholds: Dict[str, float]) -> Dict:
    per_cid: Dict[str, Dict[str, List[float]]] = {}
    for row in oof_rows:
        cid = row["cid"]
        per_cid.setdefault(cid, {"labels": [], "probs": []})
        per_cid[cid]["labels"].append(int(row["label"]))
        per_cid[cid]["probs"].append(float(row["prob_cal"]))

    per_criterion = {}
    f1s = []
    auprcs = []
    for cid, data in per_cid.items():
        ys = data["labels"]
        ps = data["probs"]
        tau = thresholds.get(cid, 0.5)
        preds = [1 if p >= tau else 0 for p in ps]
        precision, recall, f1, _ = precision_recall_fscore_support(
            ys, preds, average="binary", zero_division=0
        )
        try:
            auprc = average_precision_score(ys, ps)
        except Exception:
            auprc = 0.0
        per_criterion[cid] = {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "threshold": float(tau),
            "auprc": float(auprc),
            "n_positive": int(sum(ys)),
            "n_total": int(len(ys)),
        }
        f1s.append(f1)
        auprcs.append(auprc)

    overall = {
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else 0.0,
    }
    return {"overall": overall, "per_criterion": per_criterion}


def run_training(cfg: DictConfig) -> Dict:
    set_seed(int(_cfg_get(cfg, "seed", 42)))

    raw_dir = Path(cfg.data.raw_dir)
    output_dir = Path(_cfg_get(cfg, "output_dir", "outputs/runs/real_dev_pc_v2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_pc_examples(raw_dir)
    labels = [ex.label for ex in examples]
    cids = [ex.cid for ex in examples]
    logger.info("Loaded %d PC examples (positives=%d)", len(examples), sum(labels))

    skf = StratifiedKFold(n_splits=int(_cfg_get(cfg, "train.n_folds", 5)), shuffle=True, random_state=42)

    oof_rows: List[Dict] = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Build fold splits
        train_subset = [examples[i] for i in train_idx]
        test_subset = [examples[i] for i in test_idx]
        train_examples, eval_examples = train_test_split(
            train_subset, test_size=0.1, random_state=fold, stratify=[ex.label for ex in train_subset]
        )

        trainer, eval_ds = train_fold(fold, train_examples, eval_examples, cfg, output_dir)
        test_ds = tokenize_dataset(test_subset, trainer.tokenizer, int(_cfg_get(cfg, "model.max_length", 512)))
        logits, labels_np = predict_oof(trainer, test_ds)

        for ex, logit, label in zip(test_subset, logits, labels_np):
            oof_rows.append(
                {
                    "post_id": ex.post_id,
                    "cid": ex.cid,
                    "label": int(label),
                    "logit": float(logit),
                    "fold": fold,
                }
            )

    # Calibration on OOF
    oof_logits = [row["logit"] for row in oof_rows]
    oof_labels = [row["label"] for row in oof_rows]
    oof_cids = [row["cid"] for row in oof_rows]
    temps = fit_per_class_temperature(oof_logits, oof_labels, oof_cids)
    probs_cal = apply_temperature_to_logits(oof_logits, oof_cids, temps)
    for row, p_cal in zip(oof_rows, probs_cal):
        row["prob_cal"] = p_cal
        row["prob_uncal"] = 1 / (1 + np.exp(-row["logit"]))

    # Threshold search on calibrated OOF
    thresholds = compute_optimal_thresholds(oof_labels, probs_cal, oof_cids)
    metrics = compute_metrics_from_oof(oof_rows, thresholds)

    # Write artifacts
    oof_path = output_dir / "pc_oof.jsonl"
    with oof_path.open("w") as f:
        for row in oof_rows:
            f.write(json.dumps(row) + "\n")

    save_json(temps, output_dir / "calibration_pc.json")
    save_json(thresholds, output_dir / "thresholds_pc.json")
    save_json(metrics, output_dir / "metrics.json")

    logger.info("Wrote OOF predictions to %s", oof_path)
    return {
        "oof_path": str(oof_path),
        "calibration_path": str(output_dir / "calibration_pc.json"),
        "thresholds_path": str(output_dir / "thresholds_pc.json"),
        "metrics_path": str(output_dir / "metrics.json"),
    }
