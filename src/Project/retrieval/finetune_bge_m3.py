"""Placeholder finetuning pipeline for BGE-M3 retriever.

This script only prepares mined positive/negative pairs and writes them to disk.
Actual fine-tuning should be hooked up when gates fail, but the data artifacts
are ready for a downstream trainer to consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from Project.dataio.schemas import RetrievalSC
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.io import load_models
from Project.utils.logging import get_logger


def _collect_pairs(dataset, retrievals: List[RetrievalSC], max_neg: int = 5) -> List[dict]:
    positives = {(lab.post_id, lab.sent_id, lab.cid) for lab in dataset["labels_sc"] if lab.label == 1}
    pairs: List[dict] = []
    for ret in retrievals:
        gold_sents = {sid for (pid, sid, cid) in positives if pid == ret.post_id and cid == ret.cid}
        for sid, _score, _ranks in ret.candidates:
            label = 1 if sid in gold_sents else 0
            pairs.append(
                {
                    "post_id": ret.post_id,
                    "cid": ret.cid,
                    "sent_id": sid,
                    "label": label,
                }
            )
        # Add a few hard negatives if missing
        if not gold_sents:
            hard_negs = [sid for sid, _, _ in ret.candidates][:max_neg]
            for sid in hard_negs:
                pairs.append({"post_id": ret.post_id, "cid": ret.cid, "sent_id": sid, "label": 0})
    return pairs


def prepare_finetune_data(exp: str, bi_cfg_path: str = "configs/bi.yaml") -> Path:
    logger = get_logger(__name__)
    cfg = load_config(bi_cfg_path)
    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)
    dataset = load_raw_dataset(raw_dir)
    retrieval_path = interim_dir / "retrieval_sc.jsonl"
    if not retrieval_path.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}. Run retrieve step first.")
    retrievals: List[RetrievalSC] = load_models(retrieval_path, RetrievalSC)
    pairs = _collect_pairs(dataset, retrievals)
    out_dir = Path(cfg.get("output", f"outputs/runs/{exp}")) / "retrieval_ft"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_pairs.jsonl"
    with out_path.open("w") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")
    logger.info("Prepared %d pairs for retriever fine-tuning at %s", len(pairs), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="demo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bi_cfg", default="configs/bi.yaml")
    args = parser.parse_args()
    prepare_finetune_data(args.exp, bi_cfg_path=args.bi_cfg)
    logger = get_logger(__name__)
    logger.warning(
        "Fine-tuning hook is a placeholder. Use the prepared pairs to run an external trainer "
        "(epochs=%s, lr=%s) and rebuild indexes before re-running the sweep.",
        args.epochs,
        args.lr,
    )


if __name__ == "__main__":
    main()
