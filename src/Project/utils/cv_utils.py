"""Shared cross-validation and calibration helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from Project.calib.temperature import TemperatureScaler


def fit_per_class_temperature(
    logits: Sequence[float],
    labels: Sequence[int],
    cids: Sequence[str],
    min_samples: int = 10,
) -> Dict:
    """Fit global and per-class temperatures using logits/labels grouped by cid."""
    scaler = TemperatureScaler()
    global_temp = float(scaler.fit(logits, labels))

    cid_to_idx: Dict[str, List[int]] = defaultdict(list)
    for idx, cid in enumerate(cids):
        cid_to_idx[cid].append(idx)

    per_class_temps: Dict[str, float] = {}
    for cid, indices in cid_to_idx.items():
        cid_logits = [logits[i] for i in indices]
        cid_labels = [labels[i] for i in indices]
        if len(cid_logits) < min_samples:
            per_class_temps[cid] = global_temp
            continue
        temp_scaler = TemperatureScaler()
        per_class_temps[cid] = float(temp_scaler.fit(cid_logits, cid_labels))

    return {"global": global_temp, "per_class": per_class_temps}


def apply_temperature_to_logits(
    logits: Iterable[float],
    cids: Iterable[str],
    temps: Dict,
) -> List[float]:
    """Apply per-class temperature scaling to logits and return calibrated probs."""
    global_temp = float(temps.get("global", 1.0))
    per_class = temps.get("per_class", {})
    calibrated: List[float] = []
    for logit, cid in zip(logits, cids):
        temp = float(per_class.get(cid, global_temp))
        safe_temp = max(temp, 1e-4)
        prob = 1 / (1 + np.exp(-logit / safe_temp))
        calibrated.append(float(prob))
    return calibrated


def compute_optimal_thresholds(
    labels: Sequence[int],
    probs: Sequence[float],
    cids: Sequence[str],
) -> Dict[str, float]:
    """Find per-class thresholds maximizing F1."""
    per_cid: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for y, p, cid in zip(labels, probs, cids):
        per_cid[cid].append((int(y), float(p)))

    thresholds: Dict[str, float] = {}
    for cid, pairs in per_cid.items():
        ys, ps = zip(*pairs)
        if len(set(ys)) <= 1:
            thresholds[cid] = 1.0 if ys[0] == 0 else 0.0
            continue
        candidates = sorted(set(ps))
        best_tau, best_f1 = 0.5, -1.0
        for tau in candidates:
            preds = [1 if p >= tau else 0 for p in ps]
            f1 = f1_score(ys, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
        thresholds[cid] = best_tau
    return thresholds


def save_json(data: Dict, path: Path) -> None:
    """Write JSON to disk with indent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
