"""Inference for sentence–criterion cross-encoder."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from omegaconf import DictConfig

from Project.crossenc.common import CrossEncoder, predict
from Project.dataio.loaders import PairExample
from Project.dataio.schemas import Criterion, RetrievalSC, Sentence
from Project.metrics.calibration import ece
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.io import load_models, write_jsonl
from Project.utils.logging import get_logger


def _make_examples(
    retrievals: List[RetrievalSC],
    sentences: List[Sentence],
    criteria: List[Criterion],
    keep_topk: int,
) -> List[PairExample]:
    sent_lookup = {(s.post_id, s.sent_id): s for s in sentences}
    crit_lookup = {c.cid: c for c in criteria}
    examples: List[PairExample] = []
    for ret in retrievals:
        crit = crit_lookup.get(ret.cid)
        if crit is None:
            continue
        for cand in ret.candidates[:keep_topk]:
            if not cand:
                continue
            sent_id = cand[0]
            sent = sent_lookup.get((ret.post_id, sent_id))
            if not sent:
                continue
            examples.append(
                PairExample(
                    text=sent.text,
                    criterion=crit.desc,
                    label=0,
                    meta={"post_id": ret.post_id, "sent_id": sent.sent_id, "cid": ret.cid},
                )
            )
    return examples


def run_inference(cfg: DictConfig) -> Path:
    logger = get_logger(__name__)
    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)
    retrieval_path = interim_dir / "retrieval_sc.jsonl"
    exp = cfg.get("exp", "debug")
    out_dir = Path(cfg.get("output", f"outputs/runs/{exp}"))
    retrievals = (
        load_models(retrieval_path, RetrievalSC) if retrieval_path.exists() else []
    )
    data = load_raw_dataset(raw_dir)
    if not retrievals:
        # fallback to exhaustive pairing
        retrievals = []
        for post in data["posts"]:
            for crit in data["criteria"]:
                candidates = [(s.sent_id, 0.0) for s in data["sentences"] if s.post_id == post.post_id]
                retrievals.append(
                    RetrievalSC(post_id=post.post_id, cid=crit.cid, candidates=candidates)
                )
    examples = _make_examples(
        retrievals,
        data["sentences"],
        data["criteria"],
        keep_topk=cfg.rerank.keep_topk,
    )
    model = CrossEncoder()
    ckpt_path = Path(cfg.get("checkpoint", out_dir / "checkpoints" / "ce_sc.pt"))
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    logits = predict(model, examples, tokenizer=None, max_length=cfg.data.sent_max_len)
    calib_path = Path(cfg.get("calibration_path", out_dir / "calibration.json"))
    if calib_path.exists() and calib_path.is_file():
        import json

        T_sc = json.loads(calib_path.read_text()).get("T_sc", 1.0)
    else:
        T_sc = cfg.get("temperature", 1.0)
    probs = [float(torch.sigmoid(torch.tensor(l / T_sc))) for l in logits]
    output = []
    for ex, logit, prob in zip(examples, logits, probs):
        output.append(
            {
                "post_id": ex.meta["post_id"],
                "cid": ex.meta["cid"],
                "sent_id": ex.meta["sent_id"],
                "logit": float(logit),
                "prob": float(prob),
            }
        )
    out_path = interim_dir / "ce_sc_scores.jsonl"
    write_jsonl(out_path, output)
    logger.info("Wrote CE-SC scores to %s", out_path)
    return out_path


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    run_inference(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    main(args.cfg)
