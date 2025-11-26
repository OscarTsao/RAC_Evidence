"""Indexing utilities for BGE-M3 learned-sparse vectors.

This module keeps dependencies optional: if FlagEmbedding is unavailable it falls back
to a simple bag-of-words weighting while preserving the same interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from Project.dataio.schemas import Sentence
from Project.utils.logging import get_logger

try:  # pragma: no cover - optional dependency
    from FlagEmbedding import BGEM3FlagModel
except Exception:  # pragma: no cover
    BGEM3FlagModel = None


def _default_tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"\b\w+\b", text.lower())


def _lexical_weights_from_tokens(tokens: Iterable[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return {}
    norm = float(sum(v * v for v in counts.values())) ** 0.5 or 1.0
    return {tok: val / norm for tok, val in counts.items()}


_BGE_MODEL_CACHE = None

def _get_bge_model():
    global _BGE_MODEL_CACHE
    if _BGE_MODEL_CACHE is None and BGEM3FlagModel is not None:
        try:
            # Load model once and cache it
            _BGE_MODEL_CACHE = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        except Exception as e:
            get_logger(__name__).warning("Failed to load BGEM3FlagModel: %s", e)
    return _BGE_MODEL_CACHE


def _encode_sparse_with_bge(texts: List[str]) -> List[Dict[str, float]]:
    model = _get_bge_model()
    if model is None:  # pragma: no cover - exercised via fallback path
        return [_lexical_weights_from_tokens(_default_tokenize(t)) for t in texts]
    try:  # pragma: no cover - heavy dependency
        output = model.encode(
            texts,
            return_sparse=True,
            return_dense=False,
        )
        lexical_weights = output.get("lexical_weights", [])
        if lexical_weights:
            return lexical_weights
    except Exception as exc:  # pragma: no cover
        logger = get_logger(__name__)
        logger.warning("BGEM3 sparse encoding failed, falling back to BOW: %s", exc)
    return [_lexical_weights_from_tokens(_default_tokenize(t)) for t in texts]


@dataclass
class SparseM3Index:
    vocab: Dict[str, int]
    indices: np.ndarray
    indptr: np.ndarray
    data: np.ndarray
    sent_ids: List[str]
    post_ids: List[str]
    tokenizer: str = "default"
    _post_rows: Dict[str, List[int]] | None = None

    def __post_init__(self) -> None:
        self._post_rows = self._post_to_rows()

    def _row(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start, end = int(self.indptr[idx]), int(self.indptr[idx + 1])
        return self.indices[start:end], self.data[start:end]

    def _post_to_rows(self) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for i, post_id in enumerate(self.post_ids):
            mapping.setdefault(post_id, []).append(i)
        return mapping

    def search(self, query_weights: Dict[str, float], post_id: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Dot-product search over sparse vectors for a specific post."""
        if not query_weights:
            return []
        vocab_ids = {self.vocab[tok]: weight for tok, weight in query_weights.items() if tok in self.vocab}
        if not vocab_ids:
            return []
        post_to_rows = self._post_rows or self._post_to_rows()
        rows = post_to_rows.get(post_id, [])
        results: List[Tuple[str, float]] = []
        for row_idx in rows:
            idxs, vals = self._row(row_idx)
            score = 0.0
            for i, v in zip(idxs, vals):
                if i in vocab_ids:
                    score += float(v) * float(vocab_ids[i])
            results.append((self.sent_ids[row_idx], score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def encode_query(self, text: str) -> Dict[str, float]:
        """Encode a query using the stored vocabulary."""
        # If FlagEmbedding is available, prefer its tokenizer for better alignment.
        if BGEM3FlagModel is not None:  # pragma: no cover - optional dependency
            try:
                weights = _encode_sparse_with_bge([text])[0]
                if weights:
                    return weights
            except Exception:
                pass
        tokens = _default_tokenize(text)
        return _lexical_weights_from_tokens(tokens)


def _save_vocab(vocab: Dict[str, int], path: Path) -> None:
    tokens = [None] * len(vocab)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    path.write_text(json.dumps({"tokens": tokens}))


def _load_vocab(path: Path) -> Dict[str, int]:
    meta = json.loads(path.read_text())
    tokens: List[str] = meta.get("tokens", [])
    return {tok: i for i, tok in enumerate(tokens)}


def build_sparse_m3_index(
    sentences: Sequence[Sentence],
    output_dir: Path,
    tokenizer: str = "default",
) -> SparseM3Index:
    """Build and persist the sparse M3 index (fallback BOW when BGEM3 is unavailable)."""
    logger = get_logger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [s.text for s in sentences]
    lexical_weights = _encode_sparse_with_bge(texts)
    vocab: Dict[str, int] = {}
    indices: List[int] = []
    data: List[float] = []
    indptr: List[int] = [0]
    sent_ids: List[str] = []
    post_ids: List[str] = []
    for sent, weights in zip(sentences, lexical_weights):
        row_indices: List[int] = []
        row_data: List[float] = []
        for tok, weight in weights.items():
            idx = vocab.setdefault(tok, len(vocab))
            row_indices.append(idx)
            row_data.append(float(weight))
        if row_indices:
            order = np.argsort(row_indices)
            row_indices = [row_indices[i] for i in order]
            row_data = [row_data[i] for i in order]
        indices.extend(row_indices)
        data.extend(row_data)
        indptr.append(len(indices))
        sent_ids.append(sent.sent_id)
        post_ids.append(sent.post_id)
    np.savez_compressed(
        output_dir / "m3_sparse_index.npz",
        indices=np.array(indices, dtype=np.int64),
        indptr=np.array(indptr, dtype=np.int64),
        data=np.array(data, dtype=np.float32),
        sent_ids=np.array(sent_ids),
        post_ids=np.array(post_ids),
        tokenizer=np.array([tokenizer]),
    )
    _save_vocab(vocab, output_dir / "m3_sparse_vocab.json")
    logger.info("Saved sparse M3 index with %d sentences and %d vocab terms", len(sent_ids), len(vocab))
    return SparseM3Index(
        vocab=vocab,
        indices=np.array(indices, dtype=np.int64),
        indptr=np.array(indptr, dtype=np.int64),
        data=np.array(data, dtype=np.float32),
        sent_ids=sent_ids,
        post_ids=post_ids,
        tokenizer=tokenizer,
    )


def load_sparse_m3_index(output_dir: Path) -> SparseM3Index:
    arr = np.load(output_dir / "m3_sparse_index.npz", allow_pickle=True)
    vocab = _load_vocab(output_dir / "m3_sparse_vocab.json")
    indices = arr["indices"]
    indptr = arr["indptr"]
    data = arr["data"]
    sent_ids = [str(s) for s in arr["sent_ids"].tolist()]
    post_ids = [str(p) for p in arr["post_ids"].tolist()]
    tokenizer_arr = arr.get("tokenizer")
    tokenizer = str(tokenizer_arr.tolist()[0]) if tokenizer_arr is not None else "default"
    return SparseM3Index(
        vocab=vocab,
        indices=indices,
        indptr=indptr,
        data=data,
        sent_ids=sent_ids,
        post_ids=post_ids,
        tokenizer=tokenizer,
    )
