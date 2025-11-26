"""Unit tests for threshold_optimizer module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from Project.calib.threshold_optimizer import (
    ThresholdOptimizer,
    apply_per_class_thresholds,
    find_optimal_threshold,
    find_per_class_thresholds,
)


def test_find_optimal_threshold_balanced():
    """Test optimal threshold with balanced classes."""
    # Perfect classifier: prob = label
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.95, 0.15, 0.85, 0.1])

    threshold = find_optimal_threshold(y_true, y_prob, metric="f1")

    # Threshold should be between 0 and 1
    assert 0.0 <= threshold <= 1.0

    # Verify it produces a valid F1 score
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    assert f1 > 0.5  # With well-separated classes, F1 should be good


def test_find_optimal_threshold_imbalanced():
    """Test optimal threshold with imbalanced classes."""
    # 90% negative, 10% positive
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.2, 0.1, 0.15, 0.95])

    threshold = find_optimal_threshold(y_true, y_prob, metric="f1")

    # Threshold should optimize for imbalanced data
    assert 0.0 <= threshold <= 1.0


def test_find_optimal_threshold_no_positives():
    """Test optimal threshold when no positives exist."""
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    threshold = find_optimal_threshold(y_true, y_prob, metric="f1")

    # Should return default threshold
    assert threshold == 0.5


def test_find_optimal_threshold_all_positives():
    """Test optimal threshold when all samples are positive."""
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.6, 0.7, 0.8, 0.9])

    threshold = find_optimal_threshold(y_true, y_prob, metric="f1")

    # Should find a threshold that captures all positives
    assert 0.0 <= threshold <= 1.0


def test_find_per_class_thresholds():
    """Test per-class threshold finding."""
    predictions = [
        # Class A: 40 samples (enough for optimization)
        *[{"cid": "A", "label": 1, "prob_cal": 0.9 + i * 0.001} for i in range(20)],
        *[{"cid": "A", "label": 0, "prob_cal": 0.1 + i * 0.001} for i in range(20)],

        # Class B: 40 samples
        *[{"cid": "B", "label": 1, "prob_cal": 0.85 + i * 0.001} for i in range(20)],
        *[{"cid": "B", "label": 0, "prob_cal": 0.15 + i * 0.001} for i in range(20)],

        # Class C: 5 samples (too few, should use default)
        {"cid": "C", "label": 1, "prob_cal": 0.9},
        {"cid": "C", "label": 0, "prob_cal": 0.1},
        {"cid": "C", "label": 1, "prob_cal": 0.8},
        {"cid": "C", "label": 0, "prob_cal": 0.2},
        {"cid": "C", "label": 1, "prob_cal": 0.85},
    ]

    thresholds = find_per_class_thresholds(predictions, metric="f1", min_samples=30)

    # Should have thresholds for all classes
    assert "A" in thresholds
    assert "B" in thresholds
    assert "C" in thresholds

    # Class C should use default threshold
    assert thresholds["C"] == 0.5

    # Other thresholds should be optimized
    assert 0.0 <= thresholds["A"] <= 1.0
    assert 0.0 <= thresholds["B"] <= 1.0


def test_apply_per_class_thresholds():
    """Test applying per-class thresholds."""
    predictions = [
        {"cid": "A", "label": 1, "prob_cal": 0.7},
        {"cid": "A", "label": 0, "prob_cal": 0.3},
        {"cid": "B", "label": 1, "prob_cal": 0.8},
        {"cid": "B", "label": 0, "prob_cal": 0.2},
    ]

    thresholds = {"A": 0.6, "B": 0.75}

    y_pred = apply_per_class_thresholds(predictions, thresholds, prob_key="prob_cal")

    # Check predictions
    assert y_pred[0] == 1  # 0.7 >= 0.6
    assert y_pred[1] == 0  # 0.3 < 0.6
    assert y_pred[2] == 1  # 0.8 >= 0.75
    assert y_pred[3] == 0  # 0.2 < 0.75


def test_apply_per_class_thresholds_missing_cid():
    """Test that missing cid field raises error."""
    predictions = [
        {"label": 1, "prob_cal": 0.7},  # Missing cid
    ]

    thresholds = {"A": 0.6}

    with pytest.raises(ValueError, match="missing 'cid' field"):
        apply_per_class_thresholds(predictions, thresholds, prob_key="prob_cal")


def test_apply_per_class_thresholds_unknown_class():
    """Test applying thresholds with unknown class uses default."""
    predictions = [
        {"cid": "UNKNOWN", "label": 1, "prob_cal": 0.6},
    ]

    thresholds = {"A": 0.7}

    y_pred = apply_per_class_thresholds(predictions, thresholds, prob_key="prob_cal")

    # Should use default threshold 0.5
    assert y_pred[0] == 1  # 0.6 >= 0.5


def test_threshold_optimizer_fit_predict():
    """Test ThresholdOptimizer fit and predict."""
    optimizer = ThresholdOptimizer(metric="f1", min_samples=10)

    # Training data
    train_predictions = [
        *[{"cid": "A", "label": 1, "prob_cal": 0.9 + i * 0.001} for i in range(15)],
        *[{"cid": "A", "label": 0, "prob_cal": 0.1 + i * 0.001} for i in range(15)],
    ]

    # Fit
    thresholds = optimizer.fit(train_predictions)

    assert "A" in thresholds
    assert optimizer.thresholds_ == thresholds

    # Predict
    test_predictions = [
        {"cid": "A", "label": 1, "prob_cal": 0.95},
        {"cid": "A", "label": 0, "prob_cal": 0.05},
    ]

    y_pred = optimizer.predict(test_predictions, prob_key="prob_cal")

    assert len(y_pred) == 2
    assert y_pred[0] == 1
    assert y_pred[1] == 0


def test_threshold_optimizer_predict_before_fit():
    """Test that predict raises error before fit."""
    optimizer = ThresholdOptimizer()

    predictions = [{"cid": "A", "label": 1, "prob_cal": 0.7}]

    with pytest.raises(ValueError, match="Must call fit"):
        optimizer.predict(predictions)


def test_threshold_optimizer_save_load():
    """Test ThresholdOptimizer save and load."""
    optimizer = ThresholdOptimizer(metric="f1", min_samples=20)

    train_predictions = [
        *[{"cid": "A", "label": 1, "prob_cal": 0.9 + i * 0.001} for i in range(25)],
        *[{"cid": "A", "label": 0, "prob_cal": 0.1 + i * 0.001} for i in range(25)],
    ]

    optimizer.fit(train_predictions)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "thresholds.json"
        optimizer.save(save_path)

        # Check file exists
        assert save_path.exists()

        # Load
        loaded_optimizer = ThresholdOptimizer.load(save_path)

        # Check loaded optimizer
        assert loaded_optimizer.metric == "f1"
        assert loaded_optimizer.min_samples == 20
        assert loaded_optimizer.thresholds_ == optimizer.thresholds_

        # Verify loaded optimizer can predict
        test_predictions = [
            {"cid": "A", "label": 1, "prob_cal": 0.95},
        ]

        y_pred_original = optimizer.predict(test_predictions)
        y_pred_loaded = loaded_optimizer.predict(test_predictions)

        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)


def test_threshold_optimizer_save_creates_directory():
    """Test that save creates parent directories if needed."""
    optimizer = ThresholdOptimizer()
    optimizer.thresholds_ = {"A": 0.7}

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "nested" / "dir" / "thresholds.json"
        optimizer.save(save_path)

        assert save_path.exists()

        # Verify contents
        with open(save_path) as f:
            data = json.load(f)

        assert data["metric"] == "f1"
        assert data["min_samples"] == 30
        assert data["thresholds"] == {"A": 0.7}
