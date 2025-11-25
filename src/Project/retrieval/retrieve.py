"""Retrieve top-K sentences for each (post, criterion) using RRF fusion."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from omegaconf import DictConfig

from Project.dataio.schemas import Criterion, RetrievalSC
from Project.retrieval.index_bm25 import BM25Index
from Project.retrieval.index_faiss import PostIndex
from Project.retrieval.index_sparse_m3 import SparseM3Index
from Project.retrieval.train_bi import BiEncoderModel
from Project.utils.hydra_utils import cfg_get as _cfg_get
from Project.utils.io import dump_models
from Project.utils.logging import get_logger

ChannelRanks = Dict[str, int]
RankList = List[Tuple[str, float]]


def _channel_topk(cfg: DictConfig, channel: str, cid: str) -> int:
    topk = int(_cfg_get(cfg, f"fusion.channels.{channel}.topK", _cfg_get(cfg, "retrieve.topK", 50)))
    if _cfg_get(cfg, "fusion.dynamic_K.enabled", False):
        boosted = int(_cfg_get(cfg, "fusion.dynamic_K.boosted_topK", topk))
        boost_list = list(_cfg_get(cfg, "fusion.dynamic_K.cid_boost", []))
        if cid in boost_list:
            topk = max(topk, boosted)
    return topk


def reciprocal_rank_fusion(
    ranklists: Dict[str, RankList],
    rrf_k: int,
    final_top_k: int,
    tie_breaker: str = "dense_then_sparse",
) -> List[Tuple[str, float, ChannelRanks]]:
    """Fuse multiple ranklists with RRF and return metadata about per-channel ranks."""
    rank_maps: Dict[str, Dict[str, int]] = {
        name: {sid: rank + 1 for rank, (sid, _score) in enumerate(cands)}
        for name, cands in ranklists.items()
    }
    scores: Dict[str, float] = {}
    for name, ranks in rank_maps.items():
        for sid, rank in ranks.items():
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (rrf_k + rank)

    def _tie_key(sid: str) -> Tuple[int, int, int, str]:
        dense_rank = rank_maps.get("dense", {}).get(sid, 10**9)
        sparse_rank = rank_maps.get("sparse_m3", {}).get(sid, 10**9)
        bm25_rank = rank_maps.get("bm25", {}).get(sid, 10**9)
        if tie_breaker == "dense_then_sparse":
            return (dense_rank, sparse_rank, bm25_rank, sid)
        if tie_breaker == "dense_then_bm25":
            return (dense_rank, bm25_rank, sparse_rank, sid)
        return (dense_rank, sparse_rank, bm25_rank, sid)

    ordered = sorted(scores.keys(), key=lambda sid: (-scores[sid], _tie_key(sid)))
    fused: List[Tuple[str, float, ChannelRanks]] = []
    for sid in ordered[:final_top_k]:
        ranks = {
            "rank_d": rank_maps.get("dense", {}).get(sid),
            "rank_s": rank_maps.get("sparse_m3", {}).get(sid),
            "rank_b": rank_maps.get("bm25", {}).get(sid),
        }
        fused.append((sid, float(scores[sid]), ranks))
    return fused


class FusionRetriever:
    """Three-way fusion retriever with dense, sparse, and BM25 channels."""

    def __init__(
        self,
        model: BiEncoderModel,
        dense_index: Dict[str, PostIndex],
        cfg: DictConfig,
        sparse_index: SparseM3Index | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.model = model
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.bm25_index = bm25_index
        self.cfg = cfg
        self.logger = get_logger(__name__)
        self.cache_enabled = bool(_cfg_get(cfg, "fusion.cache.query_embed", True))
        self.bm25_cache_enabled = bool(_cfg_get(cfg, "fusion.cache.bm25_hits", True))
        self._cached_bm25: Dict[Tuple[str, str], RankList] = {}
        self._encode_query = (
            lru_cache(maxsize=256)(self._encode_query_uncached) if self.cache_enabled else self._encode_query_uncached
        )

    def _encode_query_uncached(self, cid: str, text_query: str):
        dense_vec = self.model.encode([text_query])
        sparse_weights = self.sparse_index.encode_query(text_query) if self.sparse_index else {}
        return dense_vec, sparse_weights

    def _dense_search(self, post_id: str, query_dense, top_k: int) -> RankList:
        if post_id not in self.dense_index:
            return []
        return self.dense_index[post_id].search(query_dense, top_k=top_k)

    def _sparse_search(self, post_id: str, query_weights: Dict[str, float], top_k: int) -> RankList:
        if not self.sparse_index:
            return []
        return self.sparse_index.search(query_weights, post_id=post_id, top_k=top_k)

    def _bm25_search(self, post_id: str, text_query: str, top_k: int) -> RankList:
        if not self.bm25_index:
            return []
        cache_key = (post_id, text_query)
        if self.bm25_cache_enabled and cache_key in self._cached_bm25:
            return self._cached_bm25[cache_key][:top_k]
        hits = self.bm25_index.search(text_query, post_id=post_id, top_k=top_k)
        if self.bm25_cache_enabled:
            self._cached_bm25[cache_key] = hits
        return hits

    def retrieve_candidates(
        self,
        post_id: str,
        criterion_id: str,
        text_query: str,
    ) -> List[Tuple[str, float, ChannelRanks]]:
        mode = _cfg_get(self.cfg, "fusion.mode", "dense_only")
        final_top_k = int(_cfg_get(self.cfg, "fusion.final_topK", _cfg_get(self.cfg, "retrieve.topK", 50)))
        rrf_k = int(_cfg_get(self.cfg, "fusion.rrf_k", 60))
        tie_breaker = str(_cfg_get(self.cfg, "fusion.tie_breaker", "dense_then_sparse"))
        same_post_filter = bool(_cfg_get(self.cfg, "fusion.same_post_filter", True))
        dense_topk = _channel_topk(self.cfg, "dense", criterion_id)
        sparse_topk = _channel_topk(self.cfg, "sparse_m3", criterion_id)
        bm25_topk = _channel_topk(self.cfg, "bm25", criterion_id)
        dense_enabled = bool(_cfg_get(self.cfg, "fusion.channels.dense.enabled", True))
        sparse_enabled = bool(_cfg_get(self.cfg, "fusion.channels.sparse_m3.enabled", False))
        bm25_enabled = bool(_cfg_get(self.cfg, "fusion.channels.bm25.enabled", False))
        query_dense, query_sparse = self._encode_query(criterion_id, text_query)
        dense_hits: RankList = []
        sparse_hits: RankList = []
        bm25_hits: RankList = []
        if dense_enabled:
            dense_hits = self._dense_search(post_id, query_dense, top_k=dense_topk)
        if sparse_enabled and self.sparse_index is not None:
            sparse_hits = self._sparse_search(post_id, query_sparse, top_k=sparse_topk)
        if bm25_enabled and self.bm25_index is not None:
            bm25_hits = self._bm25_search(post_id if same_post_filter else None, text_query, top_k=bm25_topk)

        ranklists: Dict[str, RankList] = {}
        if mode == "dense_only":
            if dense_hits:
                ranklists["dense"] = dense_hits
        elif mode == "native_hybrid":
            if dense_hits:
                ranklists["dense"] = dense_hits
            if sparse_hits:
                ranklists["sparse_m3"] = sparse_hits
        elif mode == "two_way_rrf":
            if dense_hits:
                ranklists["dense"] = dense_hits
            if bm25_hits:
                ranklists["bm25"] = bm25_hits
        else:  # three_way_rrf or default
            if dense_hits:
                ranklists["dense"] = dense_hits
            if sparse_hits:
                ranklists["sparse_m3"] = sparse_hits
            if bm25_hits:
                ranklists["bm25"] = bm25_hits

        if not ranklists:
            self.logger.warning("No retrieval results for post=%s cid=%s", post_id, criterion_id)
            return []
        return reciprocal_rank_fusion(ranklists, rrf_k=rrf_k, final_top_k=final_top_k, tie_breaker=tie_breaker)


def retrieve_topk(
    model: BiEncoderModel,
    index: Dict[str, PostIndex],
    criteria: Dict[str, Criterion],
    requests: Iterable[Tuple[str, str]],
    cfg: DictConfig,
    sparse_index: SparseM3Index | None = None,
    bm25_index: BM25Index | None = None,
) -> List[RetrievalSC]:
    """Retrieve candidates for a list of (post_id, cid) pairs with fusion."""
    retriever = FusionRetriever(
        model,
        index,
        cfg,
        sparse_index=sparse_index,
        bm25_index=bm25_index,
    )
    results: List[RetrievalSC] = []
    for post_id, cid in requests:
        crit = criteria.get(cid)
        if post_id not in index or crit is None:
            continue
        fused = retriever.retrieve_candidates(post_id, cid, crit.desc)
        results.append(RetrievalSC(post_id=post_id, cid=cid, candidates=fused))
    return results


def write_retrieval(
    retrievals: Sequence[RetrievalSC],
    output_path: Path,
) -> None:
    dump_models(output_path, retrievals)


def run_retrieval_pipeline(
    model: BiEncoderModel,
    index: Dict[str, PostIndex],
    criteria: Dict[str, Criterion],
    post_ids: Iterable[str],
    cfg: DictConfig,
    output_path: Path,
    sparse_index: SparseM3Index | None = None,
    bm25_index: BM25Index | None = None,
) -> List[RetrievalSC]:
    logger = get_logger(__name__)
    requests = [(pid, cid) for pid in post_ids for cid in criteria.keys()]
    start = time.time()
    results = retrieve_topk(model, index, criteria, requests, cfg, sparse_index=sparse_index, bm25_index=bm25_index)
    write_retrieval(results, output_path)
    logger.info("Wrote retrieval results to %s (%.2fs)", output_path, time.time() - start)
    return results
