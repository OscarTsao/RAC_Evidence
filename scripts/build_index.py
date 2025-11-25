"""Build sentence index for retrieval."""

from __future__ import annotations

from pathlib import Path

from Project.retrieval.index_faiss import build_per_post_index
from Project.retrieval.index_bm25 import build_bm25_index
from Project.retrieval.index_sparse_m3 import build_sparse_m3_index
from Project.retrieval.train_bi import train_bi_encoder
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger


def main(
    cfg_path: str,
    build_dense: bool | None = None,
    build_sparse_m3: bool | None = None,
    build_bm25: bool | None = None,
) -> None:
    logger = get_logger(__name__)
    cfg = load_config(cfg_path)
    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)
    exp = cfg.get("exp", "debug")
    output_dir = Path(cfg.get("output", f"outputs/runs/{exp}")) / "bi"
    dataset = load_raw_dataset(raw_dir)
    build_dense = _cfg_get(cfg, "fusion.channels.dense.enabled", True) if build_dense is None else build_dense
    build_sparse_m3 = (
        _cfg_get(cfg, "fusion.channels.sparse_m3.enabled", False) if build_sparse_m3 is None else build_sparse_m3
    )
    build_bm25 = _cfg_get(cfg, "fusion.channels.bm25.enabled", False) if build_bm25 is None else build_bm25
    model = None
    if build_dense:
        model = train_bi_encoder(dataset["sentences"], dataset["criteria"], output_dir, seed=cfg.split.seed)
        build_per_post_index(model, dataset["sentences"], interim_dir, use_faiss=cfg.faiss.get("use_faiss", True))
    if build_sparse_m3:
        sparse_dir = Path(interim_dir) / "m3_sparse"
        build_sparse_m3_index(dataset["sentences"], sparse_dir)
    if build_bm25:
        bm25_dir = Path(_cfg_get(cfg, "fusion.channels.bm25.index_dir", interim_dir / "bm25_index"))
        build_bm25_index(
            dataset["sentences"],
            bm25_dir,
            analyzer=_cfg_get(cfg, "fusion.channels.bm25.analyzer", "default"),
        )
    if not (build_dense or build_sparse_m3 or build_bm25):
        logger.warning("No index was built; check flags or config.")
    else:
        logger.info("Index ready")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--build_dense", dest="build_dense", action="store_true", help="force build dense index")
    parser.add_argument("--no-build_dense", dest="build_dense", action="store_false", help="skip dense index")
    parser.add_argument(
        "--build_sparse_m3",
        dest="build_sparse_m3",
        action="store_true",
        help="build sparse M3 index",
    )
    parser.add_argument(
        "--no-build_sparse_m3",
        dest="build_sparse_m3",
        action="store_false",
        help="skip sparse M3 index",
    )
    parser.add_argument("--build_bm25", dest="build_bm25", action="store_true", help="build BM25 index")
    parser.add_argument("--no-build_bm25", dest="build_bm25", action="store_false", help="skip BM25 index")
    parser.set_defaults(build_dense=None, build_sparse_m3=None, build_bm25=None)
    args = parser.parse_args()
    main(args.cfg, build_dense=args.build_dense, build_sparse_m3=args.build_sparse_m3, build_bm25=args.build_bm25)
