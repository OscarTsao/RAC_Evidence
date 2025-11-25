import numpy as np
from omegaconf import OmegaConf

from Project.dataio.schemas import Criterion, Sentence
from Project.retrieval.index_bm25 import build_bm25_index
from Project.retrieval.index_faiss import PostIndex
from Project.retrieval.retrieve import FusionRetriever


class DummyModel:
    def encode(self, texts):
        return np.ones((len(texts), 1), dtype=np.float32)


def test_same_post_filter_enforced(tmp_path) -> None:
    sentences = [
        Sentence(post_id="p1", sent_id="s1", text="shared token here"),
        Sentence(post_id="p2", sent_id="s2", text="shared token here"),
    ]
    dense_index = {
        "p1": PostIndex(sent_ids=["s1"], embeddings=np.array([[1.0]], dtype=np.float32)),
        "p2": PostIndex(sent_ids=["s2"], embeddings=np.array([[1.0]], dtype=np.float32)),
    }
    bm25_dir = tmp_path / "bm25_filter"
    bm25_index = build_bm25_index(sentences, bm25_dir)
    cfg = OmegaConf.create(
        {
            "retrieve": {"topK": 5},
            "fusion": {
                "mode": "three_way_rrf",
                "rrf_k": 60,
                "final_topK": 5,
                "channels": {
                    "dense": {"enabled": True, "topK": 5},
                    "sparse_m3": {"enabled": False, "topK": 5},
                    "bm25": {"enabled": True, "topK": 5, "index_dir": str(bm25_dir)},
                },
                "dynamic_K": {"enabled": False},
                "same_post_filter": True,
                "tie_breaker": "dense_then_sparse",
                "cache": {"query_embed": False, "bm25_hits": False},
            },
        }
    )
    retriever = FusionRetriever(DummyModel(), dense_index, cfg, bm25_index=bm25_index)
    crit = Criterion(cid="c1", name="c1", is_core=True, desc="shared token")
    candidates = retriever.retrieve_candidates("p1", "c1", crit.desc)
    sent_ids = [cand[0] for cand in candidates]
    assert "s2" not in sent_ids
    assert "s1" in sent_ids


def test_same_post_filter_disabled_allows_cross_post(tmp_path) -> None:
    sentences = [
        Sentence(post_id="p1", sent_id="s1", text="shared token here"),
        Sentence(post_id="p2", sent_id="s2", text="shared token here"),
    ]
    dense_index = {
        "p1": PostIndex(sent_ids=["s1"], embeddings=np.array([[1.0]], dtype=np.float32)),
        "p2": PostIndex(sent_ids=["s2"], embeddings=np.array([[1.0]], dtype=np.float32)),
    }
    bm25_dir = tmp_path / "bm25_filter"
    bm25_index = build_bm25_index(sentences, bm25_dir)
    cfg = OmegaConf.create(
        {
            "retrieve": {"topK": 5},
            "fusion": {
                "mode": "three_way_rrf",
                "rrf_k": 60,
                "final_topK": 5,
                "channels": {
                    "dense": {"enabled": False, "topK": 5},
                    "sparse_m3": {"enabled": False, "topK": 5},
                    "bm25": {"enabled": True, "topK": 5, "index_dir": str(bm25_dir)},
                },
                "dynamic_K": {"enabled": False},
                "same_post_filter": False,
                "tie_breaker": "dense_then_sparse",
                "cache": {"query_embed": False, "bm25_hits": False},
            },
        }
    )
    retriever = FusionRetriever(DummyModel(), dense_index, cfg, bm25_index=bm25_index)
    crit = Criterion(cid="c1", name="c1", is_core=True, desc="shared token")
    candidates = retriever.retrieve_candidates("p1", "c1", crit.desc)
    sent_ids = [cand[0] for cand in candidates]
    assert "s2" in sent_ids
