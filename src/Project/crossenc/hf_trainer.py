"""Hugging Face cross-encoder trainers for SC and PC using BGE reranker."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from Project.dataio.schemas import Criterion, LabelPC, LabelSC, Post, Sentence
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger


def _pair_text(a: str, b: str) -> str:
    return f"{a} [SEP] {b}"


def _build_sc_examples(
    sentences: List[Sentence],
    criteria: List[Criterion],
    labels_sc: List[LabelSC],
    neg_per_pos: int,
    seed: int,
) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    crit_desc = {c.cid: c.desc for c in criteria}
    sent_by_post: Dict[str, List[Sentence]] = {}
    for s in sentences:
        sent_by_post.setdefault(s.post_id, []).append(s)
    gold: Dict[Tuple[str, str], List[str]] = {}
    for lab in labels_sc:
        if lab.label == 1:
            gold.setdefault((lab.post_id, lab.cid), []).append(lab.sent_id)
    examples: List[Tuple[str, int]] = []
    for (post_id, cid), gold_sents in gold.items():
        desc = crit_desc.get(cid, "")
        post_sents = sent_by_post.get(post_id, [])
        gold_set = set(gold_sents)
        # positives
        for sid in gold_sents:
            sent_obj = next((s for s in post_sents if s.sent_id == sid), None)
            if sent_obj:
                examples.append((_pair_text(sent_obj.text, desc), 1))
        # negatives: sample from same post not in gold
        neg_candidates = [s for s in post_sents if s.sent_id not in gold_set]
        rng.shuffle(neg_candidates)
        for s in neg_candidates[: neg_per_pos * max(1, len(gold_sents))]:
            examples.append((_pair_text(s.text, desc), 0))
    return examples


def _build_pc_examples(
    posts: List[Post],
    criteria: List[Criterion],
    labels_pc: List[LabelPC],
) -> List[Tuple[str, int]]:
    crit_desc = {c.cid: c.desc for c in criteria}
    post_lookup = {p.post_id: p for p in posts}
    examples: List[Tuple[str, int]] = []
    for lab in labels_pc:
        post = post_lookup.get(lab.post_id)
        desc = crit_desc.get(lab.cid, "")
        if post:
            examples.append((_pair_text(post.text, desc), int(lab.label)))
    return examples


def _train_fold(
    texts: List[str],
    labels: List[int],
    cfg: DictConfig,
    output_dir: Path,
    fold: int,
) -> float:
    logger = get_logger(__name__)
    model_name = _cfg_get(cfg, "model.name", "BAAI/bge-reranker-v2-m3")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    if torch.cuda.is_available():
        model = model.cuda()  # Explicitly move model to GPU
        torch.set_float32_matmul_precision("high")
    ds = [{"text": t, "label": l} for t, l in zip(texts, labels)]
    train_ds, eval_ds = train_test_split(ds, test_size=0.1, random_state=fold, stratify=labels)

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(_cfg_get(cfg, "data.max_length", 256)),
            padding=False,
        )
        enc["labels"] = float(batch["label"])
        return enc

    import datasets

    train_ds = datasets.Dataset.from_list(train_ds).map(tokenize, batched=False, remove_columns=["text", "label"])
    eval_ds = datasets.Dataset.from_list(eval_ds).map(tokenize, batched=False, remove_columns=["text", "label"])

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    args = TrainingArguments(
        output_dir=str(output_dir / f"fold{fold}"),
        per_device_train_batch_size=int(_cfg_get(cfg, "train.batch_size", 8)),
        per_device_eval_batch_size=int(_cfg_get(cfg, "train.batch_size", 8)),
        gradient_accumulation_steps=int(_cfg_get(cfg, "train.gradient_accumulation", 1)),
        learning_rate=float(_cfg_get(cfg, "train.lr", 1.5e-5)),
        weight_decay=float(_cfg_get(cfg, "train.weight_decay", 0.01)),
        num_train_epochs=float(_cfg_get(cfg, "train.epochs", 2)),
        warmup_ratio=float(_cfg_get(cfg, "train.warmup_ratio", 0.06)),
        max_steps=int(_cfg_get(cfg, "train.max_steps", -1)),
        logging_steps=int(_cfg_get(cfg, "train.logging_steps", 50)),
        eval_strategy="steps",
        eval_steps=int(_cfg_get(cfg, "train.eval_steps", 200)),
        save_steps=int(_cfg_get(cfg, "train.save_steps", 2000)),  # Save less frequently
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        no_cuda=False,  # Explicitly enable CUDA
        bf16=bool(_cfg_get(cfg, "train.bf16", True)),
        fp16=bool(_cfg_get(cfg, "train.fp16", False)),
        gradient_checkpointing=bool(_cfg_get(cfg, "train.gradient_checkpointing", False)),  # Disable by default
        dataloader_num_workers=int(_cfg_get(cfg, "train.num_workers", 8)),  # Increase workers
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        report_to=[],
        max_grad_norm=float(_cfg_get(cfg, "train.grad_clip", 1.0)),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits.reshape(-1)))
        preds = (probs >= 0.5).astype(int)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    macro_f1 = float(metrics.get("eval_macro_f1", 0.0))
    trainer.save_model(output_dir / f"fold{fold}")
    return macro_f1


def train_ce(cfg_path: str, task: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(int(_cfg_get(cfg, "seed", 42)))
    data = load_raw_dataset(Path(cfg.data.raw_dir))
    if task == "sc":
        examples = _build_sc_examples(
            data["sentences"],
            data["criteria"],
            data["labels_sc"],
            neg_per_pos=int(_cfg_get(cfg, "data.neg_per_pos", 4)),
            seed=int(_cfg_get(cfg, "seed", 42)),
        )
    else:
        examples = _build_pc_examples(data["posts"], data["criteria"], data["labels_pc"])
    texts, labels = zip(*examples)
    texts = list(texts)
    labels = list(labels)
    logger = get_logger(__name__)
    out_dir = Path(_cfg_get(cfg, "output_dir", f"outputs/runs/{cfg.get('exp','real_dev')}/{task}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=int(_cfg_get(cfg, "seed", 42)))
    fold_metrics = []
    for fold, (train_idx, _val_idx) in enumerate(kf.split(texts)):
        fold_texts = [texts[i] for i in train_idx]
        fold_labels = [labels[i] for i in train_idx]
        f1 = _train_fold(fold_texts, fold_labels, cfg, out_dir, fold)
        fold_metrics.append({"fold": fold, "macro_f1": f1})
        logger.info("Fold %s macro_f1=%.3f", fold, f1)
    summary = {"folds": fold_metrics}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
