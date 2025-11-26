"""Typer CLI that wires together the full pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from Project.calib.temperature import run_calibration
from Project.crossenc.infer_ce_pc import run_inference as run_ce_pc_infer
from Project.crossenc.infer_ce_sc import run_inference as run_ce_sc_infer
from Project.crossenc.train_ce_pc import train_ce_pc
from Project.crossenc.train_ce_sc import train_ce_sc
from Project.graph.build_graph import build_graphs
from Project.graph.train_gnn import train_gnn
from Project.retrieval.index_faiss import build_per_post_index
from Project.retrieval.index_bm25 import build_bm25_index, load_bm25_index
from Project.retrieval.index_sparse_m3 import build_sparse_m3_index, load_sparse_m3_index
from Project.retrieval.retrieve import run_retrieval_pipeline
from Project.retrieval.train_bi import load_bi_encoder, train_bi_encoder
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger
from Project.engine.evaluate import evaluate_metrics

app = typer.Typer(help="clin-gnn-rac pipeline CLI")


@app.command()
def prepare(cfg: str = "configs/dataset.yaml") -> None:
    cfg_obj = load_config(cfg)
    logger = get_logger(__name__)
    from Project.dataio.prepare import prepare_data_splits

    prepare_data_splits(cfg)
    logger.info("Prepared dataset splits.")


@app.command()
def index(cfg: str = "configs/bi.yaml") -> None:
    cfg_obj = load_config(cfg)
    dataset = load_raw_dataset(Path(cfg_obj.data.raw_dir))
    exp = cfg_obj.get("exp", "debug")
    model_dir = Path(cfg_obj.get("output", f"outputs/runs/{exp}")) / "bi"
    model = train_bi_encoder(
        dataset["sentences"],
        dataset["criteria"],
        model_dir,
        seed=cfg_obj.split.seed,
        model_name=_cfg_get(cfg_obj, "model.name"),
    )
    build_per_post_index(
        model,
        dataset["sentences"],
        Path(cfg_obj.data.interim_dir),
        use_faiss=cfg_obj.faiss.get("use_faiss", True),
    )
    if _cfg_get(cfg_obj, "fusion.channels.sparse_m3.enabled", False):
        build_sparse_m3_index(
            dataset["sentences"],
            Path(
                _cfg_get(
                    cfg_obj,
                    "fusion.channels.sparse_m3.index_dir",
                    Path(cfg_obj.data.interim_dir) / "m3_sparse",
                )
            ),
        )
    if _cfg_get(cfg_obj, "fusion.channels.bm25.enabled", False):
        build_bm25_index(
            dataset["sentences"],
            Path(
                _cfg_get(
                    cfg_obj,
                    "fusion.channels.bm25.index_dir",
                    Path(cfg_obj.data.interim_dir) / "bm25_index",
                )
            ),
            analyzer=_cfg_get(cfg_obj, "fusion.channels.bm25.analyzer", "default"),
        )


@app.command()
def retrieve(cfg: str = "configs/bi.yaml") -> None:
    cfg_obj = load_config(cfg)
    dataset = load_raw_dataset(Path(cfg_obj.data.raw_dir))
    exp = cfg_obj.get("exp", "debug")
    model_dir = Path(cfg_obj.get("output", f"outputs/runs/{exp}")) / "bi"
    model = load_bi_encoder(model_dir)
    from Project.retrieval.index_faiss import load_per_post_index

    index_map = load_per_post_index(Path(cfg_obj.data.interim_dir) / "per_post_index.pkl")
    sparse_index = None
    bm25_index = None
    sparse_dir = Path(
        _cfg_get(
            cfg_obj,
            "fusion.channels.sparse_m3.index_dir",
            Path(cfg_obj.data.interim_dir) / "m3_sparse",
        )
    )
    bm25_dir = Path(
        _cfg_get(cfg_obj, "fusion.channels.bm25.index_dir", Path(cfg_obj.data.interim_dir) / "bm25_index")
    )
    if sparse_dir.exists():
        sparse_index = load_sparse_m3_index(sparse_dir)
    if bm25_dir.exists():
        bm25_index = load_bm25_index(bm25_dir)
    run_retrieval_pipeline(
        model,
        index_map,
        {c.cid: c for c in dataset["criteria"]},
        [p.post_id for p in dataset["posts"]],
        cfg=cfg_obj,
        output_path=Path(cfg_obj.data.interim_dir) / "retrieval_sc.jsonl",
        sparse_index=sparse_index,
        bm25_index=bm25_index,
    )


@app.command()
def train_ce_sc_cmd(cfg: str = "configs/ce_sc.yaml") -> None:  # noqa: D401
    """Train sentence–criterion cross-encoder."""
    cfg_obj = load_config(cfg)
    train_ce_sc(cfg_obj)


@app.command()
def train_ce_pc_cmd(cfg: str = "configs/ce_pc.yaml") -> None:
    cfg_obj = load_config(cfg)
    train_ce_pc(cfg_obj)


@app.command()
def calibrate(cfg: str = "configs/calibrate.yaml") -> None:
    cfg_obj = load_config(cfg)
    run_calibration(cfg_obj)


@app.command()
def infer(cfg_sc: str = "configs/ce_sc.yaml", cfg_pc: str = "configs/ce_pc.yaml") -> None:
    cfg_sc_obj = load_config(cfg_sc)
    cfg_pc_obj = load_config(cfg_pc)
    run_ce_sc_infer(cfg_sc_obj)
    run_ce_pc_infer(cfg_pc_obj)


@app.command()
def build_graph(cfg: str = "configs/graph.yaml") -> None:
    cfg_obj = load_config(cfg)
    build_graphs(cfg_obj)


@app.command()
def train_gnn_cmd(cfg: str = "configs/graph.yaml") -> None:
    cfg_obj = load_config(cfg)
    train_gnn(cfg_obj)


@app.command()
def evaluate_cmd(cfg: str = "configs/evaluate.yaml") -> None:
    evaluate_metrics(cfg)


if __name__ == "__main__":
    app()
