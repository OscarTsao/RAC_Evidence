"""Utility helpers for retrieval sweeps and deterministic behavior."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def deterministic_seed(seed: int) -> None:
    """Set deterministic seeds across common libs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_sort_ranklist(cands: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Sort candidates by (-score, sent_id) for deterministic tie-breaking."""
    return sorted(cands, key=lambda x: (-float(x[1]), str(x[0])))


@dataclass
class EnvInfo:
    python: str
    platform: str
    flagembedding: str | None
    faiss: str | None
    lucene: str | None
    cuda: str | None
    seed: int
    model_name: str | None

    def as_dict(self) -> dict:
        return {
            "python": self.python,
            "platform": self.platform,
            "flagembedding": self.flagembedding,
            "faiss": self.faiss,
            "lucene": self.lucene,
            "cuda": self.cuda,
            "seed": self.seed,
            "model_name": self.model_name,
        }


def capture_env(seed: int, model_name: str | None) -> EnvInfo:
    flagembedding_ver = None
    try:
        import FlagEmbedding  # type: ignore

        flagembedding_ver = getattr(FlagEmbedding, "__version__", None)
    except Exception:
        pass

    faiss_ver = None
    try:
        import faiss  # type: ignore

        faiss_ver = faiss.__version__ if hasattr(faiss, "__version__") else "installed"
    except Exception:
        pass

    lucene_ver = None
    try:
        from pyserini.search.lucene import LuceneSearcher  # type: ignore

        lucene_ver = getattr(LuceneSearcher, "version", None)
    except Exception:
        pass

    cuda_ver = None
    try:
        import torch

        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
    except Exception:
        pass

    return EnvInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        flagembedding=flagembedding_ver,
        faiss=faiss_ver,
        lucene=lucene_ver,
        cuda=cuda_ver,
        seed=seed,
        model_name=model_name,
    )


def write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_same_post_filter(sent_ids: Iterable[Tuple[str, str]], target_post: str) -> List[str]:
    """Filter (sent_id, post_id) pairs to enforce same-post retrieval."""
    return [sid for sid, pid in sent_ids if pid == target_post]
