"""Bi-encoder training using TF-IDF or BGE-M3 backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from Project.dataio.schemas import Criterion, Sentence
from Project.utils.logging import get_logger
from Project.utils.seed import set_seed

try:  # pragma: no cover - optional dependency
    from FlagEmbedding import BGEM3FlagModel
except Exception:  # pragma: no cover
    BGEM3FlagModel = None

BI_META_FILENAME = "bi_meta.json"
DEFAULT_BGE_MODEL = "BAAI/bge-m3"


@dataclass
class BiEncoderModel:
    vectorizer: TfidfVectorizer

    backend: str = "tfidf"
    model_name: Optional[str] = None
    _bge_model: object | None = None

    def _load_bge_model(self):
        if BGEM3FlagModel is None:  # pragma: no cover - heavy dependency
            raise ImportError(
                "FlagEmbedding is required for BGE-M3 bi-encoder. Install with `pip install '.[retrieval]'`."
            )
        if self._bge_model is None:
            # Force safetensors loading to avoid torch.load security vulnerability (CVE-2025-32434)
            import os
            os.environ["TRANSFORMERS_SAFE_LOAD"] = "1"
            self._bge_model = BGEM3FlagModel(
                self.model_name or DEFAULT_BGE_MODEL,
                use_fp16=True,
                model_kwargs={"use_safetensors": True}
            )
        return self._bge_model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "bge-m3":
            model = self._load_bge_model()
            output = model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
            )
            # FlagEmbedding returns a dict with dense_vecs; fall back to direct output otherwise.
            if isinstance(output, dict) and "dense_vecs" in output:
                dense_vecs = output["dense_vecs"]
            else:
                dense_vecs = output
            dense = np.asarray(dense_vecs, dtype=np.float32)
            if dense.ndim == 1:
                dense = dense.reshape(1, -1)
            norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
            return dense / norms

        matrix = self.vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32)
        # l2 normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        return dense / norms


def _write_meta(output_dir: Path, backend: str, model_name: Optional[str] = None) -> None:
    meta = {"backend": backend}
    if model_name:
        meta["model_name"] = model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / BI_META_FILENAME).write_text(json.dumps(meta))


def train_bi_encoder(
    sentences: Iterable[Sentence],
    criteria: Iterable[Criterion],
    output_dir: Path,
    seed: int = 42,
    max_features: int = 2048,
    model_name: str | None = None,
) -> BiEncoderModel:
    """Fit or load a bi-encoder.

    When model_name contains 'bge-m3', use the pre-trained BGE-M3 encoder (no fitting).
    Otherwise, fall back to the TF-IDF baseline.
    """
    logger = get_logger(__name__)
    set_seed(seed)
    backend = "bge-m3" if model_name and "bge-m3" in model_name.lower() else "tfidf"
    if backend == "bge-m3":
        if BGEM3FlagModel is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "FlagEmbedding is required for BGE-M3 bi-encoder. Install with `pip install '.[retrieval]'`."
            )
        _write_meta(output_dir, backend, model_name=model_name or DEFAULT_BGE_MODEL)
        logger.info("Configured BGE-M3 bi-encoder (%s)", model_name or DEFAULT_BGE_MODEL)
        return BiEncoderModel(vectorizer=TfidfVectorizer(), backend=backend, model_name=model_name)

    texts = [s.text for s in sentences] + [c.desc for c in criteria]
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vectorizer.fit(texts)
    model = BiEncoderModel(vectorizer=vectorizer, backend="tfidf")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "bi_vectorizer.joblib")
    _write_meta(output_dir, "tfidf")
    logger.info("Saved bi-encoder vectorizer to %s", output_dir)
    return model


def load_bi_encoder(model_dir: Path) -> BiEncoderModel:
    meta_path = model_dir / BI_META_FILENAME
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        backend = meta.get("backend", "tfidf")
        model_name = meta.get("model_name")
        if backend == "bge-m3":
            return BiEncoderModel(vectorizer=TfidfVectorizer(), backend="bge-m3", model_name=model_name)
    # Backward-compatibility: load legacy TF-IDF-only checkpoint.
    vec = joblib.load(model_dir / "bi_vectorizer.joblib")
    return BiEncoderModel(vectorizer=vec, backend="tfidf")
