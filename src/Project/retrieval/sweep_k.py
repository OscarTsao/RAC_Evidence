"""Sweep K for retrieval recall/coverage/latency and select best trade-off."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from Project.metrics.ranking import ndcg_at_k, recall_at_k
from Project.retrieval.index_faiss import load_per_post_index
from Project.retrieval.index_sparse_m3 import load_sparse_m3_index
from Project.retrieval.retrieve import FusionRetriever
from Project.retrieval.train_bi import load_bi_encoder
from Project.retrieval.utils import capture_env, deterministic_seed, stable_sort_ranklist, write_json
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger


def _expand_out_dir(path: str, exp: str) -> Path:
    return Path(path.replace("${EXP}", exp))


def _collect_positives(labels) -> Dict[Tuple[str, str], List[str]]:
    positives: Dict[Tuple[str, str], List[str]] = {}
    for lab in labels:
        if lab.label == 1:
            positives.setdefault((lab.post_id, lab.cid), []).append(lab.sent_id)
    return positives


def _post_sentence_counts(dense_index) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pid, idx in dense_index.items():
        counts[pid] = len(getattr(idx, "sent_ids", []))
    return counts


def _load_manifest(path: Path) -> Set[str]:
    post_ids: Set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            pid = obj.get("post_id") or obj.get("id")
            if pid:
                post_ids.add(str(pid))
        except Exception:
            continue
    return post_ids


def _filter_dataset(dataset: dict, keep_posts: Set[str]) -> dict:
    if not keep_posts:
        return dataset
    filtered = {
        "posts": [p for p in dataset["posts"] if p.post_id in keep_posts],
        "sentences": [s for s in dataset["sentences"] if s.post_id in keep_posts],
        "labels_sc": [l for l in dataset["labels_sc"] if l.post_id in keep_posts],
        "labels_pc": [l for l in dataset["labels_pc"] if l.post_id in keep_posts],
        "criteria": dataset["criteria"],
    }
    return filtered


def compute_metrics_for_k(
    retrieval_map: Dict[Tuple[str, str], List[Tuple[str, float]]],
    positives: Dict[Tuple[str, str], List[str]],
    ndcg_k: int,
    k: int,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute overall and per-criterion metrics for a given K."""
    # Aggregate per criterion
    per_criterion_hits: Dict[str, List[float]] = {}
    per_criterion_cov: Dict[str, List[float]] = {}
    per_criterion_ndcg: Dict[str, List[float]] = {}
    hits_overall: List[float] = []
    ndcg_overall: List[float] = []
    for key, gold in positives.items():
        post_id, cid = key
        retrieved = retrieval_map.get(key, [])
        rec = recall_at_k(gold, retrieved, k=k)
        cov = 1.0 if rec > 0 else 0.0
        nd = ndcg_at_k(gold, retrieved, k=min(ndcg_k, k))
        hits_overall.append(rec)
        ndcg_overall.append(nd)
        per_criterion_hits.setdefault(cid, []).append(rec)
        per_criterion_cov.setdefault(cid, []).append(cov)
        per_criterion_ndcg.setdefault(cid, []).append(nd)

    def _safe_mean(arr: List[float]) -> float:
        return float(np.mean(arr)) if arr else 0.0

    overall = {
        "recall_at_k": _safe_mean(hits_overall),
        "coverage_at_k": _safe_mean(hits_overall),
        "ndcg_at_k": _safe_mean(ndcg_overall),
    }
    per_criterion: Dict[str, Dict[str, float]] = {}
    for cid in per_criterion_hits.keys():
        per_criterion[cid] = {
            "recall_at_k": _safe_mean(per_criterion_hits[cid]),
            "coverage_at_k": _safe_mean(per_criterion_cov.get(cid, [])),
            "ndcg_at_k": _safe_mean(per_criterion_ndcg.get(cid, [])),
        }
    return overall, per_criterion


def sweep_k(
    cfg_path: str,
    bi_cfg_path: str | None = None,
    exp: str | None = None,
    dev_manifest: str | None = None,
) -> Path:
    logger = get_logger(__name__)
    sweep_cfg = load_config(cfg_path)
    bi_cfg = load_config(bi_cfg_path) if bi_cfg_path else load_config("configs/bi.yaml")
    exp_name = exp or sweep_cfg.get("exp") or bi_cfg.get("exp", "demo")
    deterministic_seed(int(_cfg_get(bi_cfg, "split.seed", 42)))
    os.environ.setdefault("EXP", exp_name)
    sweep_cfg.EXP = exp_name

    data_cfg = bi_cfg.data
    raw_dir = Path(data_cfg.raw_dir)
    interim_dir = Path(data_cfg.interim_dir)
    dataset = load_raw_dataset(raw_dir)
    if dev_manifest:
        manifest_path = Path(dev_manifest)
        keep_posts = _load_manifest(manifest_path)
        dataset = _filter_dataset(dataset, keep_posts)
        logger.info("Filtered dataset to %d posts from manifest %s", len(dataset["posts"]), manifest_path)
    positives = _collect_positives(dataset["labels_sc"])
    criteria_map = {c.cid: c for c in dataset["criteria"]}
    requests = [(p.post_id, cid) for p in dataset["posts"] for cid in criteria_map.keys()]

    model_dir = Path(bi_cfg.get("output", f"outputs/runs/{exp_name}")) / "bi"
    model = load_bi_encoder(model_dir)
    dense_index = load_per_post_index(interim_dir / "per_post_index.pkl")
    sparse_index = load_sparse_m3_index(interim_dir / "m3_sparse")

    post_counts = _post_sentence_counts(dense_index)
    ks = list(_cfg_get(sweep_cfg, "sweep.ks", []))
    repeats = int(_cfg_get(sweep_cfg, "sweep.repeats", 2))
    ndcg_k = int(_cfg_get(sweep_cfg, "sweep.measure.ndcg_k", 20))
    same_post_only = bool(_cfg_get(sweep_cfg, "sweep.same_post_only", True))

    results: List[dict] = []

    for k in ks:
        cfg_k = copy.deepcopy(bi_cfg)
        cfg_k.retrieve.topK = k
        cfg_k.fusion.final_topK = k
        if getattr(cfg_k, "fusion", None) and getattr(cfg_k.fusion, "channels", None):
            for ch in ["dense", "sparse_m3", "bm25"]:
                if hasattr(cfg_k.fusion.channels, ch):
                    setattr(cfg_k.fusion.channels[ch], "topK", k)
        retriever = FusionRetriever(
            model,
            dense_index,
            cfg_k,
            sparse_index=sparse_index,
            bm25_index=None,
            default_k=k,
            min_with_post_len=_cfg_get(bi_cfg, "retrieval.min_with_post_len", True),
            same_post_only=same_post_only,
        )

        # Cold pass to warm caches
        for post_id, cid in requests:
            crit = criteria_map.get(cid)
            if not crit:
                continue
            retriever.retrieve_candidates(post_id, cid, crit.desc)

        latencies_all: List[float] = []
        retrieval_map: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}

        def _run_pass(store_results: bool = False) -> None:
            nonlocal retrieval_map
            for post_id, cid in requests:
                crit = criteria_map.get(cid)
                if not crit:
                    continue
                start = time.perf_counter()
                candidates = retriever.retrieve_candidates(post_id, cid, crit.desc)
                elapsed = (time.perf_counter() - start) * 1000.0
                latencies_all.append(elapsed)
                if store_results:
                    retrieval_map[(post_id, cid)] = stable_sort_ranklist([(c[0], c[1]) for c in candidates][:k])

        import time

        _run_pass(store_results=False)  # cold pass ignored
        for rep in range(repeats):
            _run_pass(store_results=rep == 0)

        p95 = float(np.percentile(latencies_all, 95)) if latencies_all else 0.0
        overall, per_criterion = compute_metrics_for_k(retrieval_map, positives, ndcg_k=ndcg_k, k=k)
        results.append(
            {
                "k": k,
                "overall": {**overall, "p95_latency_ms": p95},
                "per_criterion": per_criterion,
            }
        )
        logger.info("K=%s recall=%.3f p95=%.1fms", k, overall["recall_at_k"], p95)

    artifacts_dir = _expand_out_dir(str(_cfg_get(sweep_cfg, "artifacts.out_dir")), exp_name)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sweep_json_path = artifacts_dir / str(_cfg_get(sweep_cfg, "artifacts.sweep_json"))
    write_json(sweep_json_path, {"results": results, "ks": ks})

    # Plot recall / coverage / p95
    try:
        recall_vals = [r["overall"]["recall_at_k"] for r in results]
        cov_vals = [r["overall"]["coverage_at_k"] for r in results]
        p95_vals = [r["overall"]["p95_latency_ms"] for r in results]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        axes[0].plot(ks, recall_vals, marker="o")
        axes[0].set_title("Recall@K")
        axes[0].set_xlabel("K")
        axes[0].set_ylim(0, 1.05)
        axes[1].plot(ks, cov_vals, marker="o", color="orange")
        axes[1].set_title("Coverage@K")
        axes[1].set_xlabel("K")
        axes[1].set_ylim(0, 1.05)
        axes[2].plot(ks, p95_vals, marker="o", color="green")
        axes[2].set_title("p95 Latency (ms)")
        axes[2].set_xlabel("K")
        plt.tight_layout()
        fig.savefig(artifacts_dir / str(_cfg_get(sweep_cfg, "artifacts.sweep_plot_png")))
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot sweep metrics", exc_info=True)

    env = capture_env(int(_cfg_get(bi_cfg, "split.seed", 42)), _cfg_get(bi_cfg, "model.name"))
    env_path = artifacts_dir / str(_cfg_get(sweep_cfg, "artifacts.env_json"))
    write_json(env_path, env.as_dict())
    return sweep_json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/retrieval/k_selection.yaml")
    parser.add_argument("--bi_cfg", default="configs/bi.yaml")
    parser.add_argument("--exp", default=None)
    parser.add_argument("--dev_manifest", default=None, help="Optional JSONL manifest to restrict to a split")
    args = parser.parse_args()
    sweep_k(args.cfg, bi_cfg_path=args.bi_cfg, exp=args.exp, dev_manifest=args.dev_manifest)


if __name__ == "__main__":
    main()
