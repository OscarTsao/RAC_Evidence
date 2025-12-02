"""Build heterogeneous graphs with BGE-M3 embeddings and top-K selection."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from transformers import AutoModel, AutoTokenizer

from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.io import read_jsonl
from Project.utils.logging import get_logger


def _load_ground_truth_labels(labels_path: Path) -> Dict[Tuple[str, str], int]:
    """Load ground truth PC labels from labels_pc.jsonl.

    Returns:
        Dict mapping (post_id, cid) -> label (0 or 1)
    """
    labels_data = read_jsonl(labels_path)
    labels_dict = {}
    for item in labels_data:
        key = (item["post_id"], item["cid"])
        labels_dict[key] = int(item["label"])
    return labels_dict


def _load_bge_model(model_name: str = "BAAI/bge-m3"):
    """Load BGE-M3 model for embedding extraction."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def _extract_embeddings(texts: List[str], model, tokenizer, batch_size: int = 32) -> torch.Tensor:
    """Extract BGE-M3 dense embeddings for a list of texts."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}

            outputs = model(**encoded)
            # Use [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


def _load_oof_predictions(oof_path: Path) -> Tuple[List[dict], Dict[str, dict]]:
    """Load OOF predictions and group by (post_id, cid)."""
    oof_data = read_jsonl(oof_path)

    # Group by (post_id, cid)
    grouped = defaultdict(list)
    for item in oof_data:
        key = (item["post_id"], item["cid"])
        grouped[key].append(item)

    return oof_data, grouped


def _normalize_retrieval_scores(oof_data: List[dict], retrieval_data: List[dict]) -> Tuple[StandardScaler, StandardScaler]:
    """Compute normalization parameters for retrieval scores."""
    # Extract retrieval scores from retrieval_data
    dense_scores = []
    sparse_scores = []

    retrieval_lookup = {}
    for item in retrieval_data:
        for cand in item.get("candidates", []):
            sent_id, score, ranks = cand
            key = (item["post_id"], sent_id, item["cid"])
            retrieval_lookup[key] = {
                "dense_score": score,
                "rank_d": ranks.get("rank_d", 999),
                "rank_s": ranks.get("rank_s", 999)
            }
            dense_scores.append(score)
            if ranks.get("rank_s") is not None:
                sparse_scores.append(1.0 / (ranks["rank_s"] + 1))  # Convert rank to score

    # Fit scalers
    dense_scaler = StandardScaler()
    if dense_scores:
        dense_scaler.fit(np.array(dense_scores).reshape(-1, 1))

    return dense_scaler, retrieval_lookup


def _select_top_k_edges(grouped_predictions: Dict[Tuple[str, str], List[dict]], k: int = 15) -> Dict[Tuple[str, str], List[dict]]:
    """Select top-K predictions per (post_id, cid) pair based on calibrated probability."""
    top_k_dict = {}
    for (post_id, cid), items in grouped_predictions.items():
        # Sort by calibrated probability descending
        sorted_items = sorted(items, key=lambda x: x["prob_cal"], reverse=True)
        top_k_dict[(post_id, cid)] = sorted_items[:k]
    return top_k_dict


def _build_post_graph(
    post_id: str,
    sentences: List[dict],
    criteria: List[dict],
    oof_top_k: Dict[Tuple[str, str], List[dict]],
    pc_scores: Dict[str, dict],
    sent_embeddings: torch.Tensor,
    crit_embeddings: torch.Tensor,
    dense_scaler: StandardScaler,
    retrieval_lookup: Dict,
    sent_id_to_idx: Dict[str, int],
    cid_to_idx: Dict[str, int],
    labels_pc_gt: Dict[Tuple[str, str], int],
) -> HeteroData:
    """Build heterogeneous graph for a single post with semantic embeddings."""
    data = HeteroData()

    # Get post sentences
    post_sentences = [s for s in sentences if s["post_id"] == post_id]
    num_sent = len(post_sentences)

    if num_sent == 0:
        return None

    # Node features - Sentence nodes (use BGE-M3 embeddings)
    sent_indices = [sent_id_to_idx[s["sent_id"]] for s in post_sentences]
    data["sentence"].x = sent_embeddings[sent_indices]
    data["sentence"].num_nodes = num_sent

    # Node features - Criterion nodes (use BGE-M3 embeddings)
    data["criterion"].x = crit_embeddings
    data["criterion"].num_nodes = len(criteria)

    # Node features - Post node (mean of sentence embeddings)
    data["post"].x = data["sentence"].x.mean(dim=0, keepdim=True)
    data["post"].num_nodes = 1

    # Criterion labels from ground truth
    criterion_labels = []
    for c in criteria:
        gt_key = (post_id, c['cid'])
        gt_label = labels_pc_gt.get(gt_key, 0)  # Default to 0 if not found
        criterion_labels.append(float(gt_label))
    data["criterion"].y = torch.tensor(criterion_labels, dtype=torch.float)

    # Build S-C edges (sentence -> criterion) using top-K selections
    sc_edge_src = []
    sc_edge_dst = []
    sc_edge_attr = []
    sc_edge_label = []

    for (pid, cid), predictions in oof_top_k.items():
        if pid != post_id:
            continue

        crit_idx = cid_to_idx[cid]

        for pred in predictions:
            sent_id = pred["sent_id"]
            # Find sentence index in post_sentences
            sent_idx = next((i for i, s in enumerate(post_sentences) if s["sent_id"] == sent_id), None)
            if sent_idx is None:
                continue

            # Get retrieval scores
            retr_key = (post_id, sent_id, cid)
            retr_info = retrieval_lookup.get(retr_key, {})
            dense_score = retr_info.get("dense_score", 0.0)
            rank_d = retr_info.get("rank_d")
            rank_s = retr_info.get("rank_s")

            # Normalize dense score
            dense_score_norm = dense_scaler.transform([[dense_score]])[0][0] if dense_score != 0.0 else 0.0

            # Convert ranks to normalized scores (inverse rank), handle None
            inv_rank_d = 1.0 / (rank_d + 1) if rank_d is not None else 0.0
            inv_rank_s = 1.0 / (rank_s + 1) if rank_s is not None else 0.0

            sc_edge_src.append(sent_idx)
            sc_edge_dst.append(crit_idx)
            sc_edge_attr.append([
                float(pred["logit"]),
                float(pred["prob_cal"]),
                float(dense_score_norm),
                float(inv_rank_d),
                float(inv_rank_s),
            ])
            sc_edge_label.append(1.0 if pred["label"] == 1 else 0.0)

    if sc_edge_attr:
        data[("sentence", "supports", "criterion")].edge_index = torch.tensor(
            [sc_edge_src, sc_edge_dst], dtype=torch.long
        )
        data[("sentence", "supports", "criterion")].edge_attr = torch.tensor(
            sc_edge_attr, dtype=torch.float
        )
        data[("sentence", "supports", "criterion")].edge_label = torch.tensor(
            sc_edge_label, dtype=torch.float
        )

    # Build S-S edges (sequential sentence connections)
    if num_sent > 1:
        ss_edge_src = list(range(num_sent - 1))
        ss_edge_dst = list(range(1, num_sent))
        data[("sentence", "next", "sentence")].edge_index = torch.tensor(
            [ss_edge_src, ss_edge_dst], dtype=torch.long
        )

    # Build P-S edges (post contains sentences)
    data[("post", "contains", "sentence")].edge_index = torch.tensor(
        [[0] * num_sent, list(range(num_sent))], dtype=torch.long
    )

    # Build P-C edges (post matches criterion) with ground truth labels
    pc_edge_src = []
    pc_edge_dst = []
    pc_edge_attr = []
    pc_edge_label = []

    for crit_idx, c in enumerate(criteria):
        pc_key = f"{post_id}_{c['cid']}"
        if pc_key in pc_scores:
            pc_score = pc_scores[pc_key]
            pc_edge_src.append(0)  # Single post node
            pc_edge_dst.append(crit_idx)
            pc_edge_attr.append([
                float(pc_score.get("logit", 0.0)),
                float(pc_score.get("prob_cal", 0.0))
            ])
            # Use ground truth label instead of model prediction
            gt_key = (post_id, c['cid'])
            gt_label = labels_pc_gt.get(gt_key, 0)
            pc_edge_label.append(float(gt_label))

    if pc_edge_attr:
        data[("post", "matches", "criterion")].edge_index = torch.tensor(
            [pc_edge_src, pc_edge_dst], dtype=torch.long
        )
        data[("post", "matches", "criterion")].edge_attr = torch.tensor(
            pc_edge_attr, dtype=torch.float
        )
        data[("post", "matches", "criterion")].edge_label = torch.tensor(
            pc_edge_label, dtype=torch.float
        )

    return data


def build_hetero_graphs(cfg: DictConfig) -> Path:
    """Build heterogeneous graphs with semantic embeddings and save individually."""
    logger = get_logger(__name__)

    # Load data
    raw_dir = Path(cfg.data.raw_dir)
    interim_dir = Path(cfg.data.interim_dir)
    run_dir = Path(cfg.get("output_dir", f"outputs/runs/{cfg.exp}"))

    logger.info("Loading raw dataset...")
    dataset = load_raw_dataset(raw_dir)
    posts = dataset["posts"]
    sentences = [s.model_dump() for s in dataset["sentences"]]
    criteria = [c.model_dump() for c in dataset["criteria"]]

    # Load OOF predictions
    logger.info("Loading OOF predictions...")
    # Try both reranker subdirectory and root directory
    oof_path = run_dir / "reranker" / "oof_predictions.jsonl"
    if not oof_path.exists():
        oof_path = run_dir / "oof_predictions.jsonl"
    oof_data, grouped_oof = _load_oof_predictions(oof_path)

    # Load retrieval data
    logger.info("Loading retrieval data...")
    retrieval_data = read_jsonl(interim_dir / "retrieval_sc.jsonl")

    # Normalize retrieval scores
    logger.info("Normalizing retrieval scores...")
    dense_scaler, retrieval_lookup = _normalize_retrieval_scores(oof_data, retrieval_data)

    # Select top-K edges per (post, cid)
    k = cfg.graph.edges.supports.get("topk_per_cid", 15)
    logger.info(f"Selecting top-{k} edges per (post, cid)...")
    oof_top_k = _select_top_k_edges(grouped_oof, k=k)

    # Load PC scores (if available)
    pc_scores = {}
    # Try to find PC predictions in various locations
    pc_paths_to_try = [
        run_dir / "pc" / "oof_predictions.jsonl",
        Path(f"{run_dir}_pc") / "oof_predictions.jsonl",
    ]

    for pc_path in pc_paths_to_try:
        if pc_path.exists():
            logger.info(f"Loading PC scores from {pc_path}...")
            pc_data = read_jsonl(pc_path)
            for item in pc_data:
                key = f"{item['post_id']}_{item['cid']}"
                pc_scores[key] = item
            break

    if not pc_scores:
        logger.warning("No PC scores found - will use placeholder values")

    # Load ground truth PC labels
    logger.info("Loading ground truth PC labels...")
    processed_dir = Path(cfg.data.processed_dir)
    labels_pc_path = processed_dir / "labels_pc.jsonl"
    if not labels_pc_path.exists():
        raise FileNotFoundError(f"Ground truth labels not found at {labels_pc_path}")
    labels_pc_gt = _load_ground_truth_labels(labels_pc_path)
    logger.info(f"Loaded {len(labels_pc_gt)} ground truth PC labels")

    # Extract embeddings for all sentences and criteria
    logger.info("Loading BGE-M3 model...")
    model_name = cfg.model.get("name", "BAAI/bge-m3")
    model, tokenizer = _load_bge_model(model_name)

    logger.info("Extracting sentence embeddings...")
    sent_texts = [s["text"] for s in sentences]
    sent_embeddings = _extract_embeddings(sent_texts, model, tokenizer)
    sent_id_to_idx = {s["sent_id"]: i for i, s in enumerate(sentences)}

    logger.info("Extracting criterion embeddings...")
    crit_texts = [c["desc"] for c in criteria]
    crit_embeddings = _extract_embeddings(crit_texts, model, tokenizer)
    cid_to_idx = {c["cid"]: i for i, c in enumerate(criteria)}

    # Build graphs for each post
    logger.info(f"Building {len(posts)} heterogeneous graphs...")
    graphs_dir = run_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    valid_graphs = 0
    for idx, post in enumerate(posts):
        graph = _build_post_graph(
            post.post_id,
            sentences,
            criteria,
            oof_top_k,
            pc_scores,
            sent_embeddings,
            crit_embeddings,
            dense_scaler,
            retrieval_lookup,
            sent_id_to_idx,
            cid_to_idx,
            labels_pc_gt,
        )

        if graph is not None:
            # Save individual graph
            graph_path = graphs_dir / f"data_{idx}.pt"
            torch.save(graph, graph_path)
            valid_graphs += 1

    # Save metadata
    metadata = {
        "num_graphs": valid_graphs,
        "num_posts": len(posts),
        "num_sentences": len(sentences),
        "num_criteria": len(criteria),
        "top_k": k,
        "embedding_dim": sent_embeddings.shape[1],
    }
    with open(graphs_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Built {valid_graphs} graphs and saved to {graphs_dir}")
    logger.info(f"Metadata: {metadata}")

    return graphs_dir


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    build_hetero_graphs(cfg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
