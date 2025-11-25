"""Ranking metrics for retrieval and reranking."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


def recall_at_k(
    gold_sent_ids: Iterable[str],
    retrieved: Sequence[Tuple[str, float] | Sequence[object]],
    k: int = 50,
) -> float:
    gold = set(gold_sent_ids)
    if not gold:
        return 0.0
    hits = [cand[0] for cand in retrieved[:k] if cand and cand[0] in gold]
    return 1.0 if hits else 0.0


def mrr_at_k(
    gold_sent_ids: Iterable[str],
    retrieved: Sequence[Tuple[str, float] | Sequence[object]],
    k: int = 50,
) -> float:
    gold = set(gold_sent_ids)
    for idx, cand in enumerate(retrieved[:k]):
        if not cand:
            continue
        if cand[0] in gold:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_at_k(
    gold_sent_ids: Iterable[str],
    retrieved: Sequence[Tuple[str, float] | Sequence[object]],
    k: int = 10,
) -> float:
    gold = set(gold_sent_ids)
    dcg = 0.0
    for i, cand in enumerate(retrieved[:k]):
        if not cand:
            continue
        rel = 1.0 if cand[0] in gold else 0.0
        dcg += (2**rel - 1) / math.log2(i + 2)
    ideal_rels = [1.0] * min(len(gold), k)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    if idcg == 0:
        return 0.0
    return dcg / idcg
