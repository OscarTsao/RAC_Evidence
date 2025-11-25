"""Aggregation utilities for chunk-to-post fusion."""

from __future__ import annotations

import math
from typing import Iterable, List


def aggregate(scores: Iterable[float], method: str = "max", topm: int = 2) -> float:
    """Aggregate chunk scores into a post-level score."""
    values: List[float] = list(scores)
    if not values:
        return 0.0
    if method == "max":
        return max(values)
    if method.startswith("topm"):
        m = topm
        return sum(sorted(values, reverse=True)[:m]) / m
    if method == "logsumexp":
        m = max(values)
        return m + math.log(sum(math.exp(v - m) for v in values))
    raise ValueError(f"Unknown aggregation method: {method}")
