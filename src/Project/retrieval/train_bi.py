"""Bi-encoder training using a lightweight TF-IDF backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from Project.dataio.schemas import Criterion, Sentence
from Project.utils.logging import get_logger
from Project.utils.seed import set_seed


@dataclass
class BiEncoderModel:
    vectorizer: TfidfVectorizer

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32)
        # l2 normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        return dense / norms


def train_bi_encoder(
    sentences: Iterable[Sentence],
    criteria: Iterable[Criterion],
    output_dir: Path,
    seed: int = 42,
    max_features: int = 2048,
) -> BiEncoderModel:
    """Fit a TF-IDF bi-encoder."""
    logger = get_logger(__name__)
    set_seed(seed)
    texts = [s.text for s in sentences] + [c.desc for c in criteria]
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vectorizer.fit(texts)
    model = BiEncoderModel(vectorizer=vectorizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "bi_vectorizer.joblib")
    logger.info("Saved bi-encoder vectorizer to %s", output_dir)
    return model


def load_bi_encoder(model_dir: Path) -> BiEncoderModel:
    vec = joblib.load(model_dir / "bi_vectorizer.joblib")
    return BiEncoderModel(vectorizer=vec)
