"""Reliability utilities and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from Project.metrics.calibration import ece, reliability_diagram


def plot_reliability(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    output_path: Path,
    bins: int = 10,
) -> float:
    accs, confs = reliability_diagram(y_true, y_prob, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(np.linspace(0, 1, bins, endpoint=False), accs, width=1 / bins, alpha=0.6, label="acc")
    ax.plot(np.linspace(0, 1, bins, endpoint=False), confs, marker="o", label="conf")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return ece(y_true, y_prob, bins=bins)
