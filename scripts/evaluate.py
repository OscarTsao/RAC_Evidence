"""Evaluation and retrieval ablation driver."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf

from Project.dataio.schemas import Criterion, RetrievalSC
from Project.engine.evaluate import evaluate_metrics
from Project.metrics.ranking import recall_at_k
from Project.retrieval.index_bm25 import load_bm25_index
from Project.retrieval.index_faiss import load_per_post_index
from Project.retrieval.index_sparse_m3 import load_sparse_m3_index
from Project.retrieval.retrieve import FusionRetriever
from Project.retrieval.train_bi import load_bi_encoder
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.logging import get_logger


def _collect_positives(labels_sc) -> Dict[Tuple[str, str], List[str]]:
    positives: Dict[Tuple[str, str], List[str]] = {}
    for lab in labels_sc:
        if lab.label == 1:
            positives.setdefault((lab.post_id, lab.cid), []).append(lab.sent_id)
    return positives


def _metrics(positives: Dict[Tuple[str, str], List[str]], retrievals: List[RetrievalSC]) -> Tuple[float, float]:
    recall_scores = []
    coverage_hits = 0
    for key, gold_sents in positives.items():
        post_id, cid = key
        ret = next((r for r in retrievals if r.post_id == post_id and r.cid == cid), None)
        if ret is None:
            continue
        recall_scores.append(recall_at_k(gold_sents, ret.candidates, k=50))
        if recall_at_k(gold_sents, ret.candidates, k=5) > 0:
            coverage_hits += 1
    recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    coverage = float(coverage_hits / max(len(positives), 1)) if positives else 0.0
    return recall, coverage


def _run_ablation(cfg_path: str | Path) -> Path:
    logger = get_logger(__name__)
    cfg = load_config(str(cfg_path))
    # Hydrate fusion config if evaluate config omits it
    if not hasattr(cfg, "fusion"):
        try:
            cfg.fusion = load_config("configs/bi.yaml").fusion
            logger.info("Loaded fusion defaults from configs/bi.yaml for ablation.")
        except Exception:
            cfg.fusion = OmegaConf.create(
                {
                    "mode": "three_way_rrf",
                    "rrf_k": 60,
                    "final_topK": 50,
                    "channels": {
                        "dense": {"enabled": True, "topK": 50},
                        "sparse_m3": {"enabled": False, "topK": 50},
                        "bm25": {"enabled": False, "topK": 50, "index_dir": "data/interim/bm25_index"},
                    },
                    "dynamic_K": {"enabled": False},
                    "same_post_filter": True,
                    "tie_breaker": "dense_then_sparse",
                    "cache": {"query_embed": True, "bm25_hits": True},
                }
            )
    dataset = load_raw_dataset(Path(cfg.data.raw_dir))
    interim_dir = Path(cfg.data.interim_dir)
    exp = cfg.get("exp", "debug")
    model_dir = Path(cfg.get("output", f"outputs/runs/{exp}")) / "bi"
    output_dir = Path(cfg.get("output", f"outputs/runs/{exp}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load artifacts
    model = load_bi_encoder(model_dir)
    dense_index = load_per_post_index(interim_dir / "per_post_index.pkl")
    sparse_index = None
    sparse_dir = interim_dir / "m3_sparse"
    if sparse_dir.exists():
        sparse_index = load_sparse_m3_index(sparse_dir)
    bm25_index = None
    bm25_dir = Path(cfg.fusion.channels.bm25.index_dir) if hasattr(cfg.fusion, "channels") and hasattr(cfg.fusion.channels, "bm25") else None
    if bm25_dir and bm25_dir.exists():
        bm25_index = load_bm25_index(bm25_dir)
    criteria_map: Dict[str, Criterion] = {c.cid: c for c in dataset["criteria"]}
    requests = [(p.post_id, cid) for p in dataset["posts"] for cid in criteria_map.keys()]
    positives = _collect_positives(dataset["labels_sc"])
    modes = ["dense_only", "native_hybrid", "two_way_rrf", "three_way_rrf"]
    results: Dict[str, Dict[str, float]] = {}
    baseline_recall = 0.0
    baseline_latency = 0.0
    for mode in modes:
        cfg_mode = copy.deepcopy(cfg)
        cfg_mode.fusion.mode = mode
        retriever = FusionRetriever(model, dense_index, cfg_mode, sparse_index=sparse_index, bm25_index=bm25_index)
        retrievals: List[RetrievalSC] = []
        latencies: List[float] = []
        for post_id, cid in requests:
            crit = criteria_map.get(cid)
            if not crit:
                continue
            start = time.perf_counter()
            candidates = retriever.retrieve_candidates(post_id, cid, crit.desc)
            latencies.append(time.perf_counter() - start)
            retrievals.append(RetrievalSC(post_id=post_id, cid=cid, candidates=candidates))
        recall, coverage = _metrics(positives, retrievals)
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        results[mode] = {"Recall@50": recall, "Coverage@5": coverage, "p95_latency": p95}
        if mode == "native_hybrid":
            baseline_recall = recall
            baseline_latency = p95
    ablation_path = output_dir / "retrieval_ablation.json"
    ablation_path.write_text(json.dumps(results, indent=2))
    three_way = results.get("three_way_rrf", {})
    gain = three_way.get("Recall@50", 0.0) - baseline_recall
    coverage_delta = three_way.get("Coverage@5", 0.0) - results.get("native_hybrid", {}).get("Coverage@5", 0.0)
    latency_growth = (
        (three_way.get("p95_latency", 0.0) - baseline_latency) / baseline_latency if baseline_latency else 0.0
    )
    if three_way:
        if gain < 0.03 and three_way.get("Recall@50", 0.0) < 0.90:
            logger.warning("Three-way fusion did not clear recall gate (+3pp or >=0.90). Gain=%.3f", gain)
        if coverage_delta < -1e-3:
            logger.warning("Coverage@5 dropped (delta %.3f). Consider bumping dynamic_K.", coverage_delta)
        if latency_growth > 0.30:
            logger.warning("p95 latency increased by %.1f%%", latency_growth * 100)
    logger.info("Saved retrieval ablation to %s", ablation_path)
    return ablation_path


def evaluate(cfg_path: str) -> None:
    _run_ablation(cfg_path)
    evaluate_metrics(cfg_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    evaluate(args.cfg)
