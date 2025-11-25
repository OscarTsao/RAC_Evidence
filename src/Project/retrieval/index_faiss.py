"""FAISS (or numpy) indexing for sentence embeddings."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from Project.dataio.schemas import Sentence
from Project.retrieval.train_bi import BiEncoderModel
from Project.utils.logging import get_logger

try:  # pragma: no cover - optional dependency
    import faiss
except Exception:  # pragma: no cover
    faiss = None


@dataclass
class PostIndex:
    sent_ids: List[str]
    embeddings: np.ndarray
    faiss_index: object | None = None

    def search(self, query: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        if self.faiss_index is not None:
            scores, idxs = self.faiss_index.search(query.astype(np.float32), top_k)
            return [
                (self.sent_ids[int(i)], float(scores[0][j]))
                for j, i in enumerate(idxs[0])
                if i >= 0 and j < len(self.sent_ids)
            ]
        # numpy fallback
        sims = (self.embeddings @ query.T).squeeze(axis=1)
        order = np.argsort(-sims)[:top_k]
        return [(self.sent_ids[int(i)], float(sims[i])) for i in order]


def build_per_post_index(
    model: BiEncoderModel,
    sentences: Sequence[Sentence],
    output_dir: Path,
    use_faiss: bool = True,
) -> Dict[str, PostIndex]:
    """Build per-post indexes, defaulting to a numpy fallback when FAISS is unavailable."""
    logger = get_logger(__name__)
    grouped: Dict[str, List[Sentence]] = {}
    for sent in sentences:
        grouped.setdefault(sent.post_id, []).append(sent)
    index: Dict[str, PostIndex] = {}
    for post_id, sent_list in grouped.items():
        embeddings = model.encode([s.text for s in sent_list])
        post_index = PostIndex(sent_ids=[s.sent_id for s in sent_list], embeddings=embeddings)
        if use_faiss and faiss is not None:  # pragma: no cover - environment-specific
            dim = embeddings.shape[1]
            idx = faiss.IndexFlatIP(dim)
            idx.add(embeddings.astype(np.float32))
            post_index.faiss_index = idx
        index[post_id] = post_index
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_post_index.pkl").open("wb") as f:
        pickle.dump(index, f)
    logger.info("Built index for %d posts", len(index))
    return index


def load_per_post_index(path: Path) -> Dict[str, PostIndex]:
    with path.open("rb") as f:
        return pickle.load(f)
