"""Inference for post–criterion cross-encoder with chunk aggregation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig

from Project.crossenc.common import CrossEncoder, predict
from Project.dataio.loaders import PairExample, sentence_chunker
from Project.dataio.schemas import Criterion, Post
from Project.utils.aggregation import aggregate
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.io import write_jsonl
from Project.utils.logging import get_logger


def _make_examples(
    posts: List[Post],
    criteria: List[Criterion],
    max_len: int,
    stride: int,
) -> Tuple[List[PairExample], Dict[Tuple[str, str], List[str]]]:
    crit_lookup = {c.cid: c for c in criteria}
    examples: List[PairExample] = []
    chunk_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for post in posts:
        for cid, crit in crit_lookup.items():
            chunks = sentence_chunker(post.text, max_len=max_len, stride=stride)
            for chunk in chunks:
                examples.append(
                    PairExample(
                        text=chunk["text"],
                        criterion=crit.desc,
                        label=0,
                        meta={"post_id": post.post_id, "cid": cid, "chunk_id": chunk["chunk_id"]},
                    )
                )
                chunk_index[(post.post_id, cid)].append(chunk["chunk_id"])
    return examples, chunk_index


def run_inference(cfg: DictConfig) -> Dict[str, Path]:
    logger = get_logger(__name__)
    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)
    exp = cfg.get("exp", "debug")
    out_dir = Path(cfg.get("output", f"outputs/runs/{exp}"))
    data = load_raw_dataset(raw_dir)
    examples, chunk_index = _make_examples(
        data["posts"],
        data["criteria"],
        cfg.data.chunk_max_len,
        cfg.data.stride,
    )
    model = CrossEncoder()
    ckpt_path = Path(cfg.get("checkpoint", out_dir / "checkpoints" / "ce_pc.pt"))
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    logits = predict(model, examples, tokenizer=None, max_length=cfg.data.chunk_max_len)
    calib_path = Path(cfg.get("calibration_path", out_dir / "calibration.json"))
    if calib_path.exists() and calib_path.is_file():
        import json

        T_pc = json.loads(calib_path.read_text()).get("T_pc", 1.0)
    else:
        T_pc = cfg.get("temperature", 1.0)
    probs = [float(torch.sigmoid(torch.tensor(l / T_pc))) for l in logits]
    chunk_out = []
    for ex, logit, prob in zip(examples, logits, probs):
        chunk_out.append(
            {
                "post_id": ex.meta["post_id"],
                "cid": ex.meta["cid"],
                "chunk_id": ex.meta["chunk_id"],
                "logit": float(logit),
                "prob": float(prob),
            }
        )
    chunk_path = interim_dir / "ce_pc_scores.jsonl"
    write_jsonl(chunk_path, chunk_out)
    # aggregate
    agg_method = cfg.rerank.get("agg", "max")
    post_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for record in chunk_out:
        post_scores[(record["post_id"], record["cid"])].append(record["prob"])
    post_out = []
    for (post_id, cid), scores in post_scores.items():
        agg_prob = aggregate(scores, method=agg_method, topm=cfg.rerank.get("topm", 2))
        post_out.append(
            {
                "post_id": post_id,
                "cid": cid,
                "agg": agg_method,
                "logit": float(agg_prob),
                "prob": float(agg_prob),
            }
        )
    post_path = interim_dir / "ce_pc_post.jsonl"
    write_jsonl(post_path, post_out)
    logger.info("Wrote CE-PC chunk and post scores to %s and %s", chunk_path, post_path)
    return {"chunks": chunk_path, "posts": post_path}


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    run_inference(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    main(args.cfg)
