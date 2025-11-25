from Project.retrieval.retrieve import reciprocal_rank_fusion


def test_rrf_prioritises_multi_channel_hits() -> None:
    ranklists = {
        "dense": [("a", 0.9), ("b", 0.8)],
        "sparse_m3": [("b", 0.7)],
    }
    fused = reciprocal_rank_fusion(ranklists, rrf_k=1, final_top_k=2, tie_breaker="dense_then_sparse")
    assert fused[0][0] == "b"
    assert fused[0][2]["rank_s"] == 1


def test_rrf_tie_breaker_prefers_dense() -> None:
    ranklists = {
        "dense": [("a", 0.9), ("b", 0.8)],
        "bm25": [("b", 0.9), ("a", 0.8)],
    }
    fused = reciprocal_rank_fusion(ranklists, rrf_k=0, final_top_k=2, tie_breaker="dense_then_sparse")
    assert [sid for sid, _, _ in fused] == ["a", "b"]
    assert fused[0][2]["rank_d"] == 1
    assert fused[1][2]["rank_b"] == 1
