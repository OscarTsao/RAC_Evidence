from pathlib import Path

from Project.calib.temperature import run_calibration
from Project.crossenc.infer_ce_pc import run_inference as infer_pc
from Project.crossenc.infer_ce_sc import run_inference as infer_sc
from Project.crossenc.train_ce_pc import train_ce_pc
from Project.crossenc.train_ce_sc import train_ce_sc
from Project.graph.build_graph import build_graphs
from Project.graph.train_gnn import train_gnn
from Project.retrieval.index_faiss import build_per_post_index
from Project.retrieval.retrieve import run_retrieval_pipeline
from Project.retrieval.train_bi import load_bi_encoder, train_bi_encoder
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.engine.evaluate import evaluate_metrics


def test_end_to_end_small(tmp_path_factory) -> None:
    exp = "ci"
    # Prepare splits
    dataset_cfg = load_config("configs/dataset.yaml")
    dataset_cfg.exp = exp
    from Project.dataio.prepare import prepare_data_splits

    prepare_data_splits("configs/dataset.yaml")
    data = load_raw_dataset(Path(dataset_cfg.data.raw_dir))
    # Train and index
    bi_cfg = load_config("configs/bi.yaml")
    bi_cfg.exp = exp
    bi_dir = Path(f"outputs/runs/{exp}/bi")
    model = train_bi_encoder(data["sentences"], data["criteria"], bi_dir, seed=0)
    index = build_per_post_index(model, data["sentences"], Path(bi_cfg.data.interim_dir), use_faiss=False)
    criteria_map = {c.cid: c for c in data["criteria"]}
    run_retrieval_pipeline(
        model,
        index,
        criteria_map,
        [p.post_id for p in data["posts"]],
        cfg=bi_cfg,
        output_path=Path(bi_cfg.data.interim_dir) / "retrieval_sc.jsonl",
    )
    # Cross-encoders
    ce_sc_cfg = load_config("configs/ce_sc.yaml")
    ce_sc_cfg.exp = exp
    train_ce_sc(ce_sc_cfg)
    infer_sc(ce_sc_cfg)
    ce_pc_cfg = load_config("configs/ce_pc.yaml")
    ce_pc_cfg.exp = exp
    train_ce_pc(ce_pc_cfg)
    infer_pc(ce_pc_cfg)
    # Calibrate
    calib_cfg = load_config("configs/calibrate.yaml")
    calib_cfg.exp = exp
    run_calibration(calib_cfg)
    # Graph + GNN
    graph_cfg = load_config("configs/graph.yaml")
    graph_cfg.exp = exp
    build_graphs(graph_cfg)
    train_gnn(graph_cfg)
    # Evaluate
    eval_cfg = load_config("configs/evaluate.yaml")
    eval_cfg.exp = exp
    metrics_path = evaluate_metrics(eval_cfg)
    assert Path(metrics_path).exists()
