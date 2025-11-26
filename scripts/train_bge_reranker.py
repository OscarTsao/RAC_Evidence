"""Train the BGE reranker cross-encoder and run K-sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

from Project.evidence.reranker import evaluate_k_sensitivity, train_reranker
from Project.utils.hydra_utils import load_config, cfg_get as _cfg_get


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/evidence_reranker.yaml")
    parser.add_argument("--exp", default=None)
    parser.add_argument("--runtime_cfg", default="configs/retrieval/runtime.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    exp = args.exp or cfg.get("exp", "real_dev")
    out_dir = train_reranker(args.cfg, exp=exp)
    retrieval_path = Path(_cfg_get(cfg, "data.retrieval_path", Path(cfg.data.interim_dir) / "retrieval_sc.jsonl"))
    evaluate_k_sensitivity(
        cfg_path=args.cfg,
        exp=exp,
        ks=[20, 30, 50],
        retrieval_path=retrieval_path,
        runtime_cfg_path=args.runtime_cfg,
    )


if __name__ == "__main__":
    main()
