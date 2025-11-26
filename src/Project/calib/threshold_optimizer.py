"""Threshold optimization for binary classification.

Finds optimal classification thresholds per class by maximizing F1 score
on validation data. Supports per-class thresholds for multi-label problems.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find optimal classification threshold for a single class.

    Args:
        y_true: Ground truth labels (0/1)
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1' currently supported)

    Returns:
        Optimal threshold value
    """
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.5  # Default threshold if no positives

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Calculate F1 at each threshold
    # Note: precision_recall_curve returns n+1 precision/recall values
    # but only n thresholds, so we trim the last precision/recall
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        f1_scores = np.nan_to_num(f1_scores)

    if len(f1_scores) == 0:
        return 0.5

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def find_per_class_thresholds(
    predictions: List[Dict],
    metric: str = "f1",
    min_samples: int = 30,
) -> Dict[str, float]:
    """Find optimal threshold for each class separately.

    Args:
        predictions: List of predictions with keys:
            - cid: class/criterion ID
            - label: ground truth (0/1)
            - prob or prob_cal: predicted probability
        metric: Metric to optimize
        min_samples: Minimum samples required per class

    Returns:
        Dict mapping class ID to optimal threshold
    """
    from collections import defaultdict

    # Group by class
    by_class = defaultdict(list)
    for pred in predictions:
        cid = pred["cid"]
        by_class[cid].append(pred)

    thresholds = {}

    for cid, class_preds in by_class.items():
        if len(class_preds) < min_samples:
            thresholds[cid] = 0.5  # Default for small classes
            continue

        y_true = np.array([p["label"] for p in class_preds])
        y_prob = np.array([p.get("prob_cal", p.get("prob", 0.0)) for p in class_preds])

        thresholds[cid] = find_optimal_threshold(y_true, y_prob, metric)

    return thresholds


def apply_per_class_thresholds(
    predictions: List[Dict],
    thresholds: Dict[str, float],
    prob_key: str = "prob_cal",
) -> np.ndarray:
    """Apply per-class thresholds to predictions.

    Args:
        predictions: List of predictions
        thresholds: Dict mapping class ID to threshold
        prob_key: Key to use for probability values

    Returns:
        Binary predictions (0/1) as numpy array
    """
    y_pred = []

    for pred in predictions:
        cid = pred.get("cid")
        if cid is None:
            raise ValueError(f"Prediction missing 'cid' field: {pred}")
        prob = pred.get(prob_key, pred.get("prob", 0.0))
        threshold = thresholds.get(cid, 0.5)
        y_pred.append(1 if prob >= threshold else 0)

    return np.array(y_pred)


class ThresholdOptimizer:
    """Threshold optimizer with scikit-learn-like interface.

    Learns optimal per-class thresholds on validation data and
    applies them to new predictions.
    """

    def __init__(self, metric: str = "f1", min_samples: int = 30):
        """Initialize threshold optimizer.

        Args:
            metric: Metric to optimize
            min_samples: Minimum samples per class
        """
        self.metric = metric
        self.min_samples = min_samples
        self.thresholds_: Dict[str, float] = {}

    def fit(self, predictions: List[Dict]) -> Dict[str, float]:
        """Learn optimal thresholds from predictions.

        Args:
            predictions: List of predictions with label and prob fields

        Returns:
            Dict of learned thresholds
        """
        self.thresholds_ = find_per_class_thresholds(
            predictions,
            metric=self.metric,
            min_samples=self.min_samples,
        )
        return self.thresholds_

    def predict(
        self,
        predictions: List[Dict],
        prob_key: str = "prob_cal",
    ) -> np.ndarray:
        """Apply learned thresholds to new predictions.

        Args:
            predictions: List of predictions
            prob_key: Key to use for probabilities

        Returns:
            Binary predictions
        """
        if not self.thresholds_:
            raise ValueError("Must call fit() before predict()")

        return apply_per_class_thresholds(predictions, self.thresholds_, prob_key)

    def save(self, path: str | Path) -> None:
        """Save thresholds to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {
                    "metric": self.metric,
                    "min_samples": self.min_samples,
                    "thresholds": self.thresholds_,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdOptimizer":
        """Load thresholds from JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded ThresholdOptimizer instance
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        optimizer = cls(
            metric=data.get("metric", "f1"),
            min_samples=data.get("min_samples", 30),
        )
        optimizer.thresholds_ = data["thresholds"]

        return optimizer
