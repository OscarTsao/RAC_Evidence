"""Latency measurement helpers for retrieval sweeps."""

from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, Iterable, Tuple


def time_queries(
    query_fn: Callable[[], None],
    repeats: int = 2,
    drop_first: bool = True,
) -> Dict[str, float]:
    """Run query_fn multiple times and report latency stats.

    Args:
        query_fn: zero-arg callable that runs the retrieval loop
        repeats: number of timed passes
        drop_first: whether to discard the first pass (cold start)

    Returns:
        Dict with p50, p95, count (all in milliseconds)
    """
    timings_ms = []
    for i in range(repeats + (1 if drop_first else 0)):
        start = time.perf_counter()
        query_fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if drop_first and i == 0:
            continue
        timings_ms.append(elapsed_ms)
    if not timings_ms:
        return {"p50": 0.0, "p95": 0.0, "count": 0}
    timings_ms.sort()
    p50 = statistics.median(timings_ms)
    idx_95 = int(len(timings_ms) * 0.95) - 1
    idx_95 = max(0, min(idx_95, len(timings_ms) - 1))
    p95 = timings_ms[idx_95]
    return {"p50": float(p50), "p95": float(p95), "count": len(timings_ms)}
