"""Unit tests for eval_retrieval module."""

from Project.metrics.eval_retrieval import (
    compute_retrieval_recall,
    compute_retrieval_recall_at_k,
)


def test_compute_retrieval_recall_perfect():
    """Test retrieval recall with perfect retrieval."""
    predictions = [
        # Post 1, criterion A: 2 positives, both retrieved
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 1, "prob": 0.9},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 1, "prob": 0.8},
        {"post_id": "p1", "sent_id": "s3", "cid": "A", "label": 0, "prob": 0.2},

        # Post 2, criterion B: 1 positive, retrieved
        {"post_id": "p2", "sent_id": "s1", "cid": "B", "label": 1, "prob": 0.95},
        {"post_id": "p2", "sent_id": "s2", "cid": "B", "label": 0, "prob": 0.1},
    ]

    metrics = compute_retrieval_recall(predictions)

    # Overall recall should be 1.0 (all positives retrieved)
    assert metrics["overall"]["n_gold_positives"] == 3
    assert metrics["overall"]["n_retrieved_positives"] == 3
    assert metrics["overall"]["retrieval_recall"] == 1.0

    # Per-criterion metrics
    assert metrics["per_criterion"]["A"]["n_gold_positives"] == 2
    assert metrics["per_criterion"]["A"]["retrieval_recall"] == 1.0
    assert metrics["per_criterion"]["B"]["n_gold_positives"] == 1
    assert metrics["per_criterion"]["B"]["retrieval_recall"] == 1.0


def test_compute_retrieval_recall_partial():
    """Test retrieval recall with partial retrieval."""
    # Gold labels include sentences NOT in predictions
    gold_labels = [
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 1},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 1},
        {"post_id": "p1", "sent_id": "s3", "cid": "A", "label": 1},  # Not retrieved
        {"post_id": "p2", "sent_id": "s1", "cid": "B", "label": 1},
        {"post_id": "p2", "sent_id": "s2", "cid": "B", "label": 1},  # Not retrieved
    ]

    predictions = [
        # Only retrieved s1 and s2 for p1, criterion A
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 1, "prob": 0.9},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 1, "prob": 0.8},

        # Only retrieved s1 for p2, criterion B
        {"post_id": "p2", "sent_id": "s1", "cid": "B", "label": 1, "prob": 0.95},
    ]

    metrics = compute_retrieval_recall(predictions, gold_labels)

    # Overall recall: 3/5 = 0.6
    assert metrics["overall"]["n_gold_positives"] == 5
    assert metrics["overall"]["n_retrieved_positives"] == 3
    assert metrics["overall"]["retrieval_recall"] == 0.6

    # Per-criterion: A: 2/3 = 0.667, B: 1/2 = 0.5
    assert metrics["per_criterion"]["A"]["n_gold_positives"] == 3
    assert metrics["per_criterion"]["A"]["n_retrieved_positives"] == 2
    assert abs(metrics["per_criterion"]["A"]["retrieval_recall"] - 0.6667) < 0.001

    assert metrics["per_criterion"]["B"]["n_gold_positives"] == 2
    assert metrics["per_criterion"]["B"]["n_retrieved_positives"] == 1
    assert metrics["per_criterion"]["B"]["retrieval_recall"] == 0.5


def test_compute_retrieval_recall_no_positives():
    """Test retrieval recall when no positives exist."""
    predictions = [
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 0, "prob": 0.2},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 0, "prob": 0.1},
    ]

    metrics = compute_retrieval_recall(predictions)

    # Should return 0.0 recall
    assert metrics["overall"]["n_gold_positives"] == 0
    assert metrics["overall"]["retrieval_recall"] == 0.0


def test_compute_retrieval_recall_from_predictions():
    """Test retrieval recall using labels from predictions (no gold_labels)."""
    predictions = [
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 1, "prob": 0.9},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 0, "prob": 0.3},
        {"post_id": "p2", "sent_id": "s1", "cid": "B", "label": 1, "prob": 0.95},
    ]

    metrics = compute_retrieval_recall(predictions, gold_labels=None)

    # All positives are retrieved (since we're using predictions)
    assert metrics["overall"]["n_gold_positives"] == 2
    assert metrics["overall"]["n_retrieved_positives"] == 2
    assert metrics["overall"]["retrieval_recall"] == 1.0


def test_compute_retrieval_recall_at_k():
    """Test retrieval recall@K for different K values."""
    predictions = [
        # Post 1, criterion A: 5 candidates, 2 positives
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 1, "prob": 0.95},  # rank 1
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 0, "prob": 0.85},  # rank 2
        {"post_id": "p1", "sent_id": "s3", "cid": "A", "label": 1, "prob": 0.75},  # rank 3
        {"post_id": "p1", "sent_id": "s4", "cid": "A", "label": 0, "prob": 0.65},  # rank 4
        {"post_id": "p1", "sent_id": "s5", "cid": "A", "label": 0, "prob": 0.55},  # rank 5

        # Post 2, criterion B: 3 candidates, 1 positive
        {"post_id": "p2", "sent_id": "s1", "cid": "B", "label": 0, "prob": 0.9},   # rank 1
        {"post_id": "p2", "sent_id": "s2", "cid": "B", "label": 1, "prob": 0.8},   # rank 2
        {"post_id": "p2", "sent_id": "s3", "cid": "B", "label": 0, "prob": 0.7},   # rank 3
    ]

    metrics = compute_retrieval_recall_at_k(predictions, k_values=[1, 2, 3, 5])

    # Total positives: 3 (2 in p1-A, 1 in p2-B)
    assert metrics["total_positives"] == 3

    # K=1: Only top-1 in each group
    # p1-A: s1 (positive), p2-B: s1 (negative) -> 1/3 = 0.333
    assert abs(metrics["recall@1"] - 0.3333) < 0.001

    # K=2: Top-2 in each group
    # p1-A: s1 (pos), s2 (neg), p2-B: s1 (neg), s2 (pos) -> 2/3 = 0.667
    assert abs(metrics["recall@2"] - 0.6667) < 0.001

    # K=3: Top-3 in each group
    # p1-A: s1 (pos), s2 (neg), s3 (pos), p2-B: s1 (neg), s2 (pos), s3 (neg) -> 3/3 = 1.0
    assert metrics["recall@3"] == 1.0

    # K=5: Top-5 (all for p1-A, all for p2-B)
    assert metrics["recall@5"] == 1.0


def test_compute_retrieval_recall_at_k_custom_prob_key():
    """Test retrieval recall@K with custom probability key."""
    predictions = [
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 1, "prob_cal": 0.95},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 0, "prob_cal": 0.85},
        {"post_id": "p1", "sent_id": "s3", "cid": "A", "label": 1, "prob_cal": 0.75},
    ]

    metrics = compute_retrieval_recall_at_k(predictions, k_values=[1, 2], prob_key="prob_cal")

    # Should use prob_cal for ranking
    assert abs(metrics["recall@1"] - 0.5) < 0.001  # 1 positive in top-1
    assert metrics["recall@2"] == 1.0  # Both positives in top-2


def test_compute_retrieval_recall_at_k_no_positives():
    """Test retrieval recall@K when no positives exist."""
    predictions = [
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 0, "prob": 0.9},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 0, "prob": 0.8},
    ]

    metrics = compute_retrieval_recall_at_k(predictions, k_values=[1, 5])

    assert metrics["total_positives"] == 0
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 0.0


def test_compute_retrieval_recall_at_k_multiple_groups():
    """Test retrieval recall@K with multiple (post, cid) groups."""
    predictions = [
        # Group 1: p1-A (2 positives)
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 1, "prob": 0.9},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 1, "prob": 0.8},
        {"post_id": "p1", "sent_id": "s3", "cid": "A", "label": 0, "prob": 0.7},

        # Group 2: p1-B (1 positive)
        {"post_id": "p1", "sent_id": "s1", "cid": "B", "label": 0, "prob": 0.95},
        {"post_id": "p1", "sent_id": "s2", "cid": "B", "label": 1, "prob": 0.85},

        # Group 3: p2-A (1 positive)
        {"post_id": "p2", "sent_id": "s1", "cid": "A", "label": 1, "prob": 0.99},
        {"post_id": "p2", "sent_id": "s2", "cid": "A", "label": 0, "prob": 0.88},
    ]

    metrics = compute_retrieval_recall_at_k(predictions, k_values=[1, 2])

    # Total positives: 4
    assert metrics["total_positives"] == 4

    # K=1: p1-A: s1 (pos), p1-B: s1 (neg), p2-A: s1 (pos) -> 2/4 = 0.5
    assert metrics["recall@1"] == 0.5

    # K=2: p1-A: s1,s2 (both pos), p1-B: s1,s2 (s2 pos), p2-A: s1,s2 (s1 pos) -> 4/4 = 1.0
    assert metrics["recall@2"] == 1.0


def test_compute_retrieval_recall_per_criterion_breakdown():
    """Test that per-criterion breakdown is correct."""
    predictions = [
        # Criterion A: 2 positives, 3 total retrieved
        {"post_id": "p1", "sent_id": "s1", "cid": "A", "label": 1, "prob": 0.9},
        {"post_id": "p1", "sent_id": "s2", "cid": "A", "label": 1, "prob": 0.8},
        {"post_id": "p1", "sent_id": "s3", "cid": "A", "label": 0, "prob": 0.7},

        # Criterion B: 1 positive, 2 total retrieved
        {"post_id": "p2", "sent_id": "s1", "cid": "B", "label": 1, "prob": 0.95},
        {"post_id": "p2", "sent_id": "s2", "cid": "B", "label": 0, "prob": 0.85},
    ]

    metrics = compute_retrieval_recall(predictions)

    # Check per-criterion breakdown
    assert len(metrics["per_criterion"]) == 2

    assert metrics["per_criterion"]["A"]["n_retrieved"] == 3
    assert metrics["per_criterion"]["A"]["n_gold_positives"] == 2
    assert metrics["per_criterion"]["A"]["n_retrieved_positives"] == 2
    assert metrics["per_criterion"]["A"]["retrieval_recall"] == 1.0

    assert metrics["per_criterion"]["B"]["n_retrieved"] == 2
    assert metrics["per_criterion"]["B"]["n_gold_positives"] == 1
    assert metrics["per_criterion"]["B"]["n_retrieved_positives"] == 1
    assert metrics["per_criterion"]["B"]["retrieval_recall"] == 1.0
