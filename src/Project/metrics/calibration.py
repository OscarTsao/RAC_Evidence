"""Calibration utilities."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def ece(y_true: Iterable[int], y_prob: Iterable[float], bins: int = 10) -> float:
    y_true_arr = np.array(list(y_true))
    y_prob_arr = np.array(list(y_prob))
    bin_ids = np.clip((y_prob_arr * bins).astype(int), 0, bins - 1)
    total = len(y_true_arr)
    if total == 0:
        return 0.0
    error = 0.0
    for i in range(bins):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        acc = y_true_arr[mask].mean()
        conf = y_prob_arr[mask].mean()
        error += mask.mean() * abs(acc - conf)
    return float(error)


def nll(y_true: Iterable[int], logits: Iterable[float], temperature: float = 1.0) -> float:
    y_true_arr = np.array(list(y_true))
    logits_arr = np.array(list(logits)) / max(temperature, 1e-6)
    probs = 1.0 / (1.0 + np.exp(-logits_arr))
    probs = np.clip(probs, 1e-8, 1 - 1e-8)
    loss = -(y_true_arr * np.log(probs) + (1 - y_true_arr) * np.log(1 - probs))
    return float(loss.mean())


def reliability_diagram(y_true: Iterable[int], y_prob: Iterable[float], bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
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
