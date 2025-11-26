"""Cross-encoder evidence reranker training using Hugging Face Transformers.

This uses BAAI/bge-reranker-v2-m3 with optional LoRA and evaluates macro-F1
over retrieval candidates for K-sensitivity.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from Project.dataio.schemas import RetrievalSC
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.io import load_models
from Project.utils.logging import get_logger


def _pair_text(sentence: str, criterion_desc: str) -> str:
    return f"{sentence} [SEP] {criterion_desc}"


def _positives(labels_sc) -> Dict[Tuple[str, str], List[str]]:
    pos: Dict[Tuple[str, str], List[str]] = {}
    for lab in labels_sc:
        if lab.label == 1:
            pos.setdefault((lab.post_id, lab.cid), []).append(lab.sent_id)
    return pos


def _load_retrieval(path: Path) -> List[RetrievalSC]:
    return load_models(path, RetrievalSC)


def _build_examples(
    dataset: dict,
    retrievals: List[RetrievalSC],
    top_k_train: int,
    neg_per_pos: int,
    seed: int,
) -> Tuple[List[str], List[int]]:
    rng = random.Random(seed)
    crit_desc = {c.cid: c.desc for c in dataset["criteria"]}
    pos_map = _positives(dataset["labels_sc"])
    ret_map: Dict[Tuple[str, str], List[str]] = {
        (r.post_id, r.cid): [sid for sid, _score, _r in r.candidates[:top_k_train]] for r in retrievals
    }
    texts: List[str] = []
    labels: List[int] = []
    for (post_id, cid), gold_sents in pos_map.items():
        desc = crit_desc.get(cid, "")
        for sid in gold_sents:
            sent_obj = next((s for s in dataset["sentences"] if s.post_id == post_id and s.sent_id == sid), None)
            if not sent_obj:
                continue
            texts.append(_pair_text(sent_obj.text, desc))
            labels.append(1)
        cand_list = ret_map.get((post_id, cid), [])
        cand_negs = [sid for sid in cand_list if sid not in gold_sents]
        rng.shuffle(cand_negs)
        cand_negs = cand_negs[: neg_per_pos * max(1, len(gold_sents))]
        if len(cand_negs) < neg_per_pos * max(1, len(gold_sents)):
            same_post = [s.sent_id for s in dataset["sentences"] if s.post_id == post_id and s.sent_id not in gold_sents]
            rng.shuffle(same_post)
            for sid in same_post:
                if sid in cand_negs:
                    continue
                cand_negs.append(sid)
                if len(cand_negs) >= neg_per_pos * max(1, len(gold_sents)):
                    break
        for sid in cand_negs:
            sent_obj = next((s for s in dataset["sentences"] if s.post_id == post_id and s.sent_id == sid), None)
            if not sent_obj:
                continue
            texts.append(_pair_text(sent_obj.text, desc))
            labels.append(0)
    return texts, labels


def build_dataset(
    cfg: DictConfig,
    retrieval_path: Path,
    seed: int,
) -> Dataset:
    dataset = load_raw_dataset(Path(cfg.data.raw_dir))
    retrievals = _load_retrieval(retrieval_path)
    texts, labels = _build_examples(
        dataset,
        retrievals,
        top_k_train=int(_cfg_get(cfg, "train.top_k_train", 100)),
        neg_per_pos=int(_cfg_get(cfg, "train.neg_per_pos", 4)),
        seed=seed,
    )
    return Dataset.from_dict({"text": texts, "label": labels})


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits.reshape(-1)))
    preds = (probs >= 0.5).astype(int)
    from sklearn.metrics import f1_score

    return {"macro_f1": f1_score(labels, preds, average="macro")}


def train_reranker(cfg_path: str, exp: str | None = None) -> Path:
    cfg = load_config(cfg_path)
    exp_name = exp or cfg.get("exp", "real_dev")
    seed = int(_cfg_get(cfg, "seed", 42))
    set_seed(seed)
    logger = get_logger(__name__)

    retrieval_path = Path(_cfg_get(cfg, "data.retrieval_path", Path(cfg.data.interim_dir) / "retrieval_sc.jsonl"))
    ds = build_dataset(cfg, retrieval_path, seed=seed)
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=0.1, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    model_name = _cfg_get(cfg, "model.name", "BAAI/bge-reranker-v2-m3")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if _cfg_get(cfg, "model.lora.enabled", False):
        lora_cfg = LoraConfig(
            r=int(_cfg_get(cfg, "model.lora.r", 16)),
            lora_alpha=int(_cfg_get(cfg, "model.lora.alpha", 32)),
            lora_dropout=float(_cfg_get(cfg, "model.lora.dropout", 0.1)),
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "dense"],
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_cfg)
        logger.info("Enabled LoRA r=%s alpha=%s", lora_cfg.r, lora_cfg.lora_alpha)

    def _tokenize(batch):
        enc = tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=int(_cfg_get(cfg, "model.max_length", 384)),
        )
        enc["labels"] = batch["label"]
        return enc

    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(_tokenize, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    output_dir = Path(_cfg_get(cfg, "output_dir", f"outputs/runs/{exp_name}/reranker"))
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(_cfg_get(cfg, "train.batch_size", 8)),
        per_device_eval_batch_size=int(_cfg_get(cfg, "train.batch_size", 8)),
        gradient_accumulation_steps=int(_cfg_get(cfg, "train.gradient_accumulation", 4)),
        learning_rate=float(_cfg_get(cfg, "train.lr", 1.5e-5)),
        weight_decay=float(_cfg_get(cfg, "train.weight_decay", 0.01)),
        num_train_epochs=float(_cfg_get(cfg, "train.epochs", 2)),
        warmup_ratio=float(_cfg_get(cfg, "train.warmup_ratio", 0.06)),
        max_steps=_cfg_get(cfg, "train.max_steps"),
        logging_steps=int(_cfg_get(cfg, "train.logging_steps", 50)),
        evaluation_strategy="steps",
        eval_steps=int(_cfg_get(cfg, "train.eval_steps", 500)),
        save_steps=int(_cfg_get(cfg, "train.save_steps", 500)),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=bool(_cfg_get(cfg, "train.bf16", True)),
        fp16=bool(_cfg_get(cfg, "train.fp16", False)),
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=[],
        max_grad_norm=float(_cfg_get(cfg, "train.grad_clip", 1.0)),
    )

    def compute_loss(model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        tokenizer=tokenizer,
        compute_loss=compute_loss,
    )
    trainer.train()
    trainer.save_model(output_dir / "best")
    tokenizer.save_pretrained(output_dir / "best")
    return output_dir


def evaluate_k_sensitivity(
    cfg_path: str,
    exp: str,
    ks: Sequence[int],
    retrieval_path: Path,
    runtime_cfg_path: str,
) -> Path:
    cfg = load_config(cfg_path)
    exp_name = exp or cfg.get("exp", "real_dev")
    output_dir = Path(_cfg_get(cfg, "output_dir", f"outputs/runs/{exp_name}/reranker"))
    best_dir = output_dir / "best"
    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_dir, num_labels=1)
    if torch.cuda.is_available():
        model = model.cuda()
        torch.set_float32_matmul_precision("high")
    dataset = load_raw_dataset(Path(cfg.data.raw_dir))
    retrievals = _load_retrieval(retrieval_path)
    crit_desc = {c.cid: c.desc for c in dataset["criteria"]}
    pos_map = _positives(dataset["labels_sc"])
    results = []
    for k in ks:
        y_true: List[int] = []
        y_prob: List[float] = []
        for r in retrievals:
            gold = set(pos_map.get((r.post_id, r.cid), []))
            desc = crit_desc.get(r.cid, "")
            pairs = []
            labels = []
            for sid, _score, _rank in r.candidates[:k]:
                sent_obj = next((s for s in dataset["sentences"] if s.post_id == r.post_id and s.sent_id == sid), None)
                if not sent_obj:
                    continue
                pairs.append(_pair_text(sent_obj.text, desc))
                labels.append(1 if sid in gold else 0)
            if not pairs:
                continue
            batch = tokenizer(pairs, padding=True, truncation=True, max_length=int(_cfg_get(cfg, "model.max_length", 384)), return_tensors="pt")
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits.view(-1).cpu()
                probs = torch.sigmoid(logits).numpy().tolist()
            y_true.extend(labels)
            y_prob.extend(probs)
        preds = [1 if p >= 0.5 else 0 for p in y_prob]
        from sklearn.metrics import f1_score

        macro_f1 = f1_score(y_true, preds, average="macro") if y_true else 0.0
        results.append({"K": k, "macro_f1": macro_f1})
    out_path = output_dir / "final_k_choice.json"
    best = max(results, key=lambda x: x["macro_f1"])
    chosen = min([r for r in results if (best["macro_f1"] - r["macro_f1"]) * 100 <= 0.5], key=lambda x: x["K"])
    out_path.write_text(json.dumps({"best": best, "chosen": chosen, "all": results}, indent=2))
    # update runtime
    import yaml

    rt = yaml.safe_load(Path(runtime_cfg_path).read_text())
    rt.setdefault("retrieval", {})
    rt["retrieval"]["k_infer_default"] = int(chosen["K"])
    Path(runtime_cfg_path).write_text(yaml.safe_dump(rt, sort_keys=False))
    return out_path
