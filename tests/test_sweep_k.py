import json
from pathlib import Path

from Project.retrieval.sweep_k import compute_metrics_for_k


def test_compute_metrics_basic(tmp_path: Path):
    positives = {
        ("p1", "c1"): ["s1"],
        ("p2", "c1"): ["s2"],
        ("p3", "c2"): ["s3"],
    }
    retrieval_map = {
        ("p1", "c1"): [("s1", 1.0), ("s9", 0.5)],
        ("p2", "c1"): [("s5", 1.0)],
        ("p3", "c2"): [("s3", 0.9)],
    }
    overall, per_criterion = compute_metrics_for_k(retrieval_map, positives, ndcg_k=5, k=5)
    assert overall["recall_at_k"] == pytest.approx(2 / 3)
    assert per_criterion["c1"]["recall_at_k"] == pytest.approx(0.5)  # average over two examples
    assert per_criterion["c2"]["recall_at_k"] == pytest.approx(1.0)
import pytest
