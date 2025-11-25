"""Build heterogeneous graphs from CE outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig

from Project.features.negation import hedge_flags, negation_flags
from Project.features.temporal import approximate_duration_days
from Project.graph.common import HeteroData
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.io import read_jsonl
from Project.utils.logging import get_logger


def _load_scores(interim_dir: Path) -> Tuple[List[dict], List[dict]]:
    sc_scores = read_jsonl(interim_dir / "ce_sc_scores.jsonl")
    pc_scores = read_jsonl(interim_dir / "ce_pc_post.jsonl") if (interim_dir / "ce_pc_post.jsonl").exists() else []
    return sc_scores, pc_scores


def _build_post_graph(
    post_id: str,
    sentences: List[dict],
    criteria: List[dict],
    sc_scores: List[dict],
    pc_scores: List[dict],
) -> HeteroData:
    data = HeteroData()
    sent_lookup = {(s["post_id"], s["sent_id"]): s for s in sentences}
    crit_lookup = {c["cid"]: c for c in criteria}
    post_sentences = [s for s in sentences if s["post_id"] == post_id]
    num_sent = len(post_sentences)
    data["post"].x = torch.ones((1, 4))
    data["post"].num_nodes = 1
    data["sentence"].x = torch.stack(
        [torch.tensor([float(len(s["text"])), 0.0, 0.0, 0.0]) for s in post_sentences]
    ) if num_sent > 0 else torch.zeros((0, 4))
    data["sentence"].num_nodes = num_sent
    criterion_features = []
    criterion_labels = []
    for c in criteria:
        criterion_features.append(
            torch.tensor([1.0 if c["is_core"] else 0.0, len(c["desc"]) / 100.0])
        )
        pc_prob = next(
            (score["prob"] for score in pc_scores if score["post_id"] == post_id and score["cid"] == c["cid"]),
            0.0,
        )
        criterion_labels.append(1.0 if pc_prob >= 0.5 else 0.0)
    data["criterion"].x = torch.stack(criterion_features)
    data["criterion"].y = torch.tensor(criterion_labels, dtype=torch.float)
    data["criterion"].num_nodes = len(criteria)
    # post -> sentence edges
    if num_sent > 0:
        edge_index = torch.tensor([[0] * num_sent, list(range(num_sent))], dtype=torch.long)
        data[("post", "contains", "sentence")].edge_index = edge_index
    # sentence -> criterion edges
    sc_by_post = [s for s in sc_scores if s["post_id"] == post_id]
    sc_edge_index_src: List[int] = []
    sc_edge_index_dst: List[int] = []
    sc_edge_attr: List[List[float]] = []
    for score in sc_by_post:
        sent = sent_lookup.get((score["post_id"], score["sent_id"]))
        crit = crit_lookup.get(score["cid"])
        if sent is None or crit is None:
            continue
        sent_idx = next((i for i, s in enumerate(post_sentences) if s["sent_id"] == score["sent_id"]), None)
        if sent_idx is None:
            continue
        crit_idx = next((i for i, c in enumerate(criteria) if c["cid"] == score["cid"]), None)
        if crit_idx is None:
            continue
        sc_edge_index_src.append(sent_idx)
        sc_edge_index_dst.append(crit_idx)
        neg_flag = 1.0 if any(negation_flags(sent["text"])) else 0.0
        hedge_flag = 1.0 if any(hedge_flags(sent["text"])) else 0.0
        duration = approximate_duration_days(sent["text"]) or 0
        pos_norm = sent_idx / max(1, num_sent - 1) if num_sent > 1 else 0.0
        sc_edge_attr.append(
            [
                float(score["logit"]),
                float(score["prob"]),
                neg_flag,
                hedge_flag,
                float(duration),
                float(pos_norm),
            ]
        )
    if sc_edge_attr:
        data[("sentence", "supports", "criterion")].edge_index = torch.tensor(
            [sc_edge_index_src, sc_edge_index_dst], dtype=torch.long
        )
        data[("sentence", "supports", "criterion")].edge_attr = torch.tensor(
            sc_edge_attr, dtype=torch.float
        )
        data[("sentence", "supports", "criterion")].edge_label = torch.tensor(
            [1 if attr[1] >= 0.5 else 0 for attr in sc_edge_attr], dtype=torch.float
        )
    # post -> criterion edges (matches)
    pc_by_post = [p for p in pc_scores if p["post_id"] == post_id]
    if pc_by_post:
        match_index_src: List[int] = []
        match_index_dst: List[int] = []
        match_attr: List[List[float]] = []
        for score in pc_by_post:
            crit_idx = next((i for i, c in enumerate(criteria) if c["cid"] == score["cid"]), None)
            if crit_idx is None:
                continue
            match_index_src.append(0)  # single post node
            match_index_dst.append(crit_idx)
            match_attr.append([float(score["logit"]), float(score["prob"])])
        data[("post", "matches", "criterion")].edge_index = torch.tensor(
            [match_index_src, match_index_dst], dtype=torch.long
        )
        data[("post", "matches", "criterion")].edge_attr = torch.tensor(match_attr, dtype=torch.float)
        data[("post", "matches", "criterion")].edge_label = torch.tensor(
            [1 if attr[1] >= 0.5 else 0 for attr in match_attr], dtype=torch.float
        )
    return data


def build_graphs(cfg: DictConfig) -> Path:
    logger = get_logger(__name__)
    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)
    processed_dir = Path(cfg.data.processed_dir)
    sc_scores, pc_scores = _load_scores(interim_dir)
    dataset = load_raw_dataset(raw_dir)
    posts = dataset["posts"]
    sentences = [s.model_dump() for s in dataset["sentences"]]
    criteria = [c.model_dump() for c in dataset["criteria"]]
    graphs: List[HeteroData] = []
    for post in posts:
        graph = _build_post_graph(post.post_id, sentences, criteria, sc_scores, pc_scores)
        graphs.append(graph)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "graphs.pt"
    torch.save(graphs, output_path)
    logger.info("Built %d graphs and saved to %s", len(graphs), output_path)
    return output_path


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    build_graphs(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
