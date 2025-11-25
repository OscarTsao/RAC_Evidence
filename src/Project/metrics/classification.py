"""Classification metrics."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def binary_pr_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> dict:
    y_true_arr = np.array(list(y_true))
    y_prob_arr = np.array(list(y_prob))
    y_pred = (y_prob_arr >= threshold).astype(int)
    return {
        "f1": f1_score(y_true_arr, y_pred),
        "precision": None if len(y_pred) == 0 else float(
            (y_pred[y_true_arr == 1] == 1).sum() / max((y_pred == 1).sum(), 1)
        ),
        "recall": None if len(y_pred) == 0 else float(
            (y_true_arr[y_pred == 1] == 1).sum() / max((y_true_arr == 1).sum(), 1)
        ),
        "auprc": average_precision_score(y_true_arr, y_prob_arr),
    }


def macro_micro_f1(
    y_true: List[List[int]],
    y_pred: List[List[int]],
) -> Tuple[float, float]:
    """Compute macro and micro F1 across criteria."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return float(macro_f1), float(micro_f1)


def auroc_macro(y_true: List[List[int]], y_prob: List[List[float]]) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob, average="macro"))
    except ValueError:
        return 0.0


def precision_at_k(y_true_ids: Iterable[str], ranked_ids: List[str], k: int) -> float:
    positives = set(y_true_ids)
    if not positives:
        return 0.0
    hits = [sid for sid in ranked_ids[:k] if sid in positives]
    return len(hits) / k


def coverage_at_k(y_true_ids: Iterable[str], ranked_ids: List[str], k: int) -> float:
    positives = set(y_true_ids)
    if not positives:
        return 0.0
    hits = set(ranked_ids[:k]) & positives
    return len(hits) / len(positives)


def reliability_curve(y_true: Iterable[int], y_prob: Iterable[float], bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Return bin accuracies and confidences for reliability plotting."""
    y_true_arr = np.array(list(y_true))
    y_prob_arr = np.array(list(y_prob))
    bin_ids = np.clip((y_prob_arr * bins).astype(int), 0, bins - 1)
    accs = np.zeros(bins)
    confs = np.zeros(bins)
    counts = np.zeros(bins)
    for i in range(bins):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        counts[i] = mask.sum()
        accs[i] = y_true_arr[mask].mean()
        confs[i] = y_prob_arr[mask].mean()
    return accs, confs
