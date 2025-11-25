"""Train cross-encoder for sentence–criterion binding."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from omegaconf import DictConfig

from Project.crossenc.common import TrainingConfig, train_model
from Project.dataio.loaders import PairExample
from Project.dataio.schemas import Criterion, Sentence
from Project.utils import enable_performance_optimizations
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config, save_config
from Project.utils.logging import get_logger


def _build_examples(sentences: List[Sentence], criteria: List[Criterion], labels_sc) -> List[PairExample]:
    sent_lookup = {(s.post_id, s.sent_id): s for s in sentences}
    crit_lookup = {c.cid: c for c in criteria}
    examples: List[PairExample] = []
    for lab in labels_sc:
        sent = sent_lookup.get((lab.post_id, lab.sent_id))
        crit = crit_lookup.get(lab.cid)
        if not sent or not crit:
            continue
        examples.append(
            PairExample(
                text=sent.text,
                criterion=crit.desc,
                label=lab.label,
                meta={"post_id": lab.post_id, "sent_id": lab.sent_id, "cid": lab.cid},
            )
        )
    return examples


def train_ce_sc(cfg: DictConfig) -> Path:
    # Enable PyTorch performance optimizations
    enable_performance_optimizations()

    logger = get_logger(__name__)
    raw_dir = Path(cfg.data.raw_dir)
    exp = cfg.get("exp", "debug")
    out_dir = Path(cfg.get("output", f"outputs/runs/{exp}"))
    data = load_raw_dataset(raw_dir)
    examples = _build_examples(data["sentences"], data["criteria"], data["labels_sc"])
    amp_dtype = str(
        getattr(cfg.train, "amp_dtype", "bf16" if cfg.train.get("fp16", False) else "fp32")
    ).lower()
    use_amp = bool(
        getattr(cfg.train, "use_amp", cfg.train.get("fp16", False) or amp_dtype in ("bf16", "fp16"))
    )
    train_cfg = TrainingConfig(
        lr=cfg.train.lr,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        max_length=cfg.data.sent_max_len,
        seed=cfg.get("seed", 42),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    model = train_model(examples, train_cfg)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ce_sc.pt"
    torch.save(model.state_dict(), ckpt_path)
    save_config(cfg, out_dir)
    logger.info("Saved CE-SC checkpoint to %s", ckpt_path)
    return ckpt_path


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    train_ce_sc(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    main(args.cfg)
