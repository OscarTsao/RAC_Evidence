"""Retrieve top-K candidates per (post, criterion)."""

from __future__ import annotations

# Force safetensors loading to avoid torch.load security vulnerability (CVE-2025-32434)
import os
os.environ["TRANSFORMERS_SAFE_LOAD"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from pathlib import Path

from Project.retrieval.index_bm25 import load_bm25_index
from Project.retrieval.index_faiss import load_per_post_index
from Project.retrieval.index_sparse_m3 import load_sparse_m3_index
from Project.retrieval.retrieve import run_retrieval_pipeline
from Project.retrieval.train_bi import load_bi_encoder
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger


def main(
    cfg_path: str,
    fusion_mode: str | None = None,
    rrf_k: int | None = None,
    final_topk: int | None = None,
    runtime_path: str | None = "configs/retrieval/runtime.yaml",
    ignore_overrides: bool = False,
) -> None:
    logger = get_logger(__name__)
    cfg = load_config(cfg_path)
    runtime_cfg = load_config(runtime_path) if runtime_path and Path(runtime_path).exists() else None
    runtime_overrides = {} if ignore_overrides else (_cfg_get(runtime_cfg, "retrieval.k_infer_overrides", {}) if runtime_cfg else {})
    runtime_default_k = None if ignore_overrides else (_cfg_get(runtime_cfg, "retrieval.k_infer_default") if runtime_cfg else None)
    min_with_post_len = bool(_cfg_get(runtime_cfg, "retrieval.min_with_post_len", True)) if runtime_cfg else True
    if fusion_mode:
        cfg.fusion.mode = fusion_mode
    if rrf_k is not None:
        cfg.fusion.rrf_k = rrf_k
    if final_topk is not None:
        cfg.fusion.final_topK = final_topk
        runtime_default_k = final_topk
    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)
    exp = cfg.get("exp", "debug")
    model_dir = Path(cfg.get("output", f"outputs/runs/{exp}")) / "bi"
    dataset = load_raw_dataset(raw_dir)
    model = load_bi_encoder(model_dir)
    index = load_per_post_index(interim_dir / "per_post_index.pkl")
    sparse_index = None
    if getattr(cfg.fusion.channels, "sparse_m3", None):
        sparse_dir = interim_dir / "m3_sparse"
        if sparse_dir.exists():
            sparse_index = load_sparse_m3_index(sparse_dir)
    bm25_index = None
    if getattr(cfg.fusion.channels, "bm25", None):
        bm25_dir = Path(cfg.fusion.channels.bm25.index_dir)
        if bm25_dir.exists():
            bm25_index = load_bm25_index(bm25_dir)
    criteria_map = {c.cid: c for c in dataset["criteria"]}
    run_retrieval_pipeline(
        model,
        index,
        criteria_map,
        [p.post_id for p in dataset["posts"]],
        cfg=cfg,
        output_path=interim_dir / "retrieval_sc.jsonl",
        sparse_index=sparse_index,
        bm25_index=bm25_index,
        k_overrides=runtime_overrides,
        default_k=runtime_default_k,
        min_with_post_len=min_with_post_len,
        same_post_only=True,
    )
    logger.info("Retrieval step finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--fusion.mode", dest="fusion_mode", help="Override fusion mode")
    parser.add_argument("--fusion.rrf_k", dest="rrf_k", type=int, help="Override RRF k")
    parser.add_argument(
        "--fusion.final_topK",
        dest="final_topk",
        type=int,
        help="Override final topK",
    )
    parser.add_argument("--runtime", dest="runtime_path", default="configs/retrieval/runtime.yaml", help="Runtime K config")
    parser.add_argument("--ignore_overrides", action="store_true", help="Ignore runtime K overrides")
    args = parser.parse_args()
    main(
        args.cfg,
        fusion_mode=args.fusion_mode,
        rrf_k=args.rrf_k,
        final_topk=args.final_topk,
        runtime_path=args.runtime_path,
        ignore_overrides=args.ignore_overrides,
    )
