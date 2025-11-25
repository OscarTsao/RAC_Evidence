"""Lightweight BM25 indexer with optional Pyserini backend."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from Project.dataio.schemas import Sentence
from Project.utils.logging import get_logger

try:  # pragma: no cover - optional dependency
    from pyserini.index.lucene import SimpleIndexer
    from pyserini.search.lucene import LuceneSearcher
except Exception:  # pragma: no cover
    SimpleIndexer = None
    LuceneSearcher = None


def _default_tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class BM25Index:
    sentences: Dict[str, str]
    post_map: Dict[str, str]
    tokenized: Dict[str, List[str]]
    doc_freq: Dict[str, int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75
    analyzer: str = "default"
    backend: str = "python"
    index_dir: Path | None = None
    _searcher: object | None = None

    def _load_searcher(self) -> None:
        if self.backend != "pyserini" or not self.index_dir or LuceneSearcher is None:
            return
        try:  # pragma: no cover - external dependency
            self._searcher = LuceneSearcher(str(self.index_dir))
        except Exception as exc:
            logger = get_logger(__name__)
            logger.warning("Failed to load LuceneSearcher, falling back to python BM25: %s", exc)
            self.backend = "python"
            self._searcher = None

    def search(self, query: str, post_id: str | None, top_k: int = 50) -> List[Tuple[str, float]]:
        if self._searcher is None and self.backend == "pyserini":
            self._load_searcher()
        if self._searcher is not None:  # pragma: no cover - exercised when pyserini is present
            hits = self._searcher.search(query, k=top_k * 3)
            results: List[Tuple[str, float]] = []
            for h in hits:
                try:
                    stored = json.loads(h.raw)
                    sid = stored.get("id") or stored.get("sent_id")
                    post = stored.get("post_id")
                except Exception:
                    sid = h.docid
                    post = None
                if post_id and post and post != post_id:
                    continue
                results.append((sid, float(h.score)))
                if len(results) >= top_k:
                    break
            return results
        return self._python_search(query, post_id, top_k)

    def _python_search(self, query: str, post_id: str | None, top_k: int) -> List[Tuple[str, float]]:
        if not self.sentences:
            return []
        query_tokens = _default_tokenize(query)
        q_freq: Dict[str, int] = {}
        for tok in query_tokens:
            q_freq[tok] = q_freq.get(tok, 0) + 1
        scores: List[Tuple[str, float]] = []
        N = len(self.sentences)
        for sent_id, tokens in self.tokenized.items():
            if post_id and self.post_map.get(sent_id) != post_id:
                continue
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            dl = len(tokens) or 1
            doc_score = 0.0
            for term, q_tf in q_freq.items():
                if term not in tf:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                denom = tf[term] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                doc_score += idf * ((tf[term] * (self.k1 + 1)) / max(denom, 1e-6)) * q_tf
            scores.append((sent_id, doc_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "sentences": self.sentences,
            "post_map": self.post_map,
            "tokenized": self.tokenized,
            "doc_freq": self.doc_freq,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "analyzer": self.analyzer,
            "backend": self.backend,
        }
        (index_dir / "bm25_fallback.json").write_text(json.dumps(meta))


def _try_build_pyserini(sentences: Sequence[Sentence], index_dir: Path) -> bool:
    if SimpleIndexer is None:  # pragma: no cover - optional dependency
        return False
    logger = get_logger(__name__)
    def _add_doc(indexer, doc: dict) -> None:
        payload = json.dumps(doc)
        if hasattr(indexer, "add_document"):
            indexer.add_document(payload)
        elif hasattr(indexer, "add_json"):
            indexer.add_json(payload)
        elif hasattr(indexer, "add_raw_json"):
            indexer.add_raw_json(payload)
        else:
            raise AttributeError("SimpleIndexer missing document ingestion method")
    try:  # pragma: no cover
        indexer = SimpleIndexer(str(index_dir))
        if hasattr(indexer, "set_opt"):
            indexer.set_opt("storeRaw", True)
        for sent in sentences:
            doc = json.dumps({"id": sent.sent_id, "contents": sent.text, "post_id": sent.post_id})
            _add_doc(indexer, json.loads(doc))
        indexer.close()
        logger.info("Built Pyserini BM25 index at %s", index_dir)
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("Pyserini build failed, falling back to python BM25: %s", exc)
        return False


def build_bm25_index(
    sentences: Sequence[Sentence],
    index_dir: Path,
    analyzer: str = "default",
) -> BM25Index:
    """Build a BM25 index; uses Pyserini when available and falls back to a pure python scorer."""
    logger = get_logger(__name__)
    index_dir.mkdir(parents=True, exist_ok=True)
    backend = "python"
    if _try_build_pyserini(sentences, index_dir):
        backend = "pyserini"
    tokenized: Dict[str, List[str]] = {}
    doc_freq: Dict[str, int] = {}
    sentences_map: Dict[str, str] = {}
    post_map: Dict[str, str] = {}
    lengths: List[int] = []
    for sent in sentences:
        tokens = _default_tokenize(sent.text)
        tokenized[sent.sent_id] = tokens
        lengths.append(len(tokens))
        sentences_map[sent.sent_id] = sent.text
        post_map[sent.sent_id] = sent.post_id
        for tok in set(tokens):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    avgdl = float(sum(lengths) / len(lengths)) if lengths else 0.0
    bm25 = BM25Index(
        sentences=sentences_map,
        post_map=post_map,
        tokenized=tokenized,
        doc_freq=doc_freq,
        avgdl=avgdl,
        analyzer=analyzer,
        backend=backend,
        index_dir=index_dir if backend == "pyserini" else None,
    )
    bm25.save(index_dir)
    logger.info("Saved BM25 index (%s backend) with %d docs", backend, len(sentences_map))
    return bm25


def load_bm25_index(index_dir: Path) -> BM25Index:
    path = index_dir / "bm25_fallback.json"
    meta = json.loads(path.read_text())
    bm25 = BM25Index(
        sentences=meta["sentences"],
        post_map=meta["post_map"],
        tokenized={k: list(v) for k, v in meta["tokenized"].items()},
        doc_freq={k: int(v) for k, v in meta["doc_freq"].items()},
        avgdl=float(meta["avgdl"]),
        k1=float(meta.get("k1", 1.5)),
        b=float(meta.get("b", 0.75)),
        analyzer=meta.get("analyzer", "default"),
        backend=meta.get("backend", "python"),
        index_dir=index_dir if meta.get("backend") == "pyserini" else None,
    )
    return bm25
