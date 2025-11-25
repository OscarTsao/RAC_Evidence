import copy
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from Project.dataio.schemas import Criterion, Sentence
from Project.metrics.ranking import recall_at_k
from Project.retrieval.index_bm25 import build_bm25_index
from Project.retrieval.index_faiss import PostIndex
from Project.retrieval.retrieve import FusionRetriever


class DummyModel:
    def encode(self, texts):
        return np.ones((len(texts), 1), dtype=np.float32)


def _base_cfg(bm25_dir: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "retrieve": {"topK": 1},
            "fusion": {
                "mode": "native_hybrid",
                "rrf_k": 60,
                "final_topK": 2,
                "channels": {
                    "dense": {"enabled": True, "topK": 1},
                    "sparse_m3": {"enabled": False, "topK": 1},
                    "bm25": {"enabled": True, "topK": 1, "index_dir": str(bm25_dir)},
                },
                "dynamic_K": {"enabled": False},
                "same_post_filter": True,
                "tie_breaker": "dense_then_sparse",
                "cache": {"query_embed": False, "bm25_hits": False},
            },
        }
    )


def test_three_way_rrf_improves_recall(tmp_path) -> None:
    sentences = [
        Sentence(post_id="p1", sent_id="s1", text="neutral filler text"),
        Sentence(post_id="p1", sent_id="s2", text="banana signal target"),
    ]
    # Dense channel ranks s1 above s2 so recall@2=0 without BM25
    dense_index = {"p1": PostIndex(sent_ids=["s1", "s2"], embeddings=np.array([[0.2], [0.1]], dtype=np.float32))}
    bm25_dir = tmp_path / "bm25"
    bm25_index = build_bm25_index(sentences, bm25_dir)
    cfg_native = _base_cfg(bm25_dir)
    criterion = Criterion(cid="c1", name="c1", is_core=True, desc="banana")
    baseline = FusionRetriever(DummyModel(), dense_index, cfg_native, bm25_index=bm25_index)
    baseline_hits = baseline.retrieve_candidates("p1", "c1", criterion.desc)
    baseline_recall = recall_at_k(["s2"], baseline_hits, k=2)
    cfg_three = copy.deepcopy(cfg_native)
    cfg_three.fusion.mode = "three_way_rrf"
    three_way = FusionRetriever(DummyModel(), dense_index, cfg_three, bm25_index=bm25_index)
    fused_hits = three_way.retrieve_candidates("p1", "c1", criterion.desc)
    recall_three = recall_at_k(["s2"], fused_hits, k=2)
    assert baseline_recall == 0.0
    assert recall_three == 1.0


def test_bm25_missing_channel_falls_back(tmp_path) -> None:
    sentences = [
        Sentence(post_id="p1", sent_id="s1", text="token one"),
        Sentence(post_id="p1", sent_id="s2", text="token two"),
    ]
    dense_index = {"p1": PostIndex(sent_ids=["s1", "s2"], embeddings=np.array([[0.3], [0.1]], dtype=np.float32))}
    bm25_dir = tmp_path / "bm25_empty"
    bm25_index = build_bm25_index(sentences, bm25_dir)

    class EmptyBM25:
        def search(self, query, post_id=None, top_k=50):
            return []

    cfg = _base_cfg(bm25_dir)
    cfg.fusion.mode = "three_way_rrf"
    retriever = FusionRetriever(DummyModel(), dense_index, cfg, bm25_index=EmptyBM25())
    hits = retriever.retrieve_candidates("p1", "c1", "token")
    assert hits, "retrieval should fall back to dense hits when bm25 is empty"
    assert hits[0][0] == "s1"
