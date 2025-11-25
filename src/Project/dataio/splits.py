"""Utilities for creating post-level splits."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from Project.dataio.schemas import Post


def split_posts(
    posts: Sequence[Post],
    ratios: Dict[str, float],
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split post_ids into train/dev/test buckets."""
    rng = random.Random(seed)
    ids = [p.post_id for p in posts]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios.get("train", 0.8))
    n_dev = int(n * ratios.get("dev", 0.1))
    train_ids = ids[:n_train]
    dev_ids = ids[n_train : n_train + n_dev]
    test_ids = ids[n_train + n_dev :]
    return {"train": train_ids, "dev": dev_ids, "test": test_ids}


def mask_by_split(
    post_ids: Sequence[str],
    split_ids: Dict[str, List[str]],
    split: str,
) -> List[str]:
    """Return ids that belong to the requested split."""
    allowed = set(split_ids.get(split, []))
    return [pid for pid in post_ids if pid in allowed]
