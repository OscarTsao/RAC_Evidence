"""Fold-aware retrieval filter for strict same-post pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.io import read_jsonl, write_jsonl
from Project.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_candidate(entry) -> Tuple[str, float]:
    if isinstance(entry, dict):
        sent_id = str(entry.get("sent_id"))
        score = float(entry.get("score", 0.0))
    else:
        # Tuple/List formats: (sent_id, score, meta)
        sent_id = str(entry[0])
        score = float(entry[1])
    return sent_id, score


def _sort_candidates(
    candidates: Sequence[Tuple[str, float]], tie_breaker: str
) -> List[Tuple[str, float]]:
    if tie_breaker == "score_then_id":
        key_fn = lambda item: (-item[1], item[0])
    else:
        # Default fallback: deterministic ordering
        key_fn = lambda item: (-item[1], item[0])
    return sorted(candidates, key=key_fn)


def _load_post_ids(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [
                item["post_id"] if isinstance(item, dict) else str(item)
                for item in data
            ]
        if isinstance(data, dict) and "posts" in data:
            return [str(pid) for pid in data["posts"]]
        if isinstance(data, dict) and "post_id" in data:
            return [str(data["post_id"])]
    except json.JSONDecodeError:
        pass

    records = read_jsonl(path)
    if records:
        post_ids: List[str] = []
        for rec in records:
            if isinstance(rec, dict) and "post_id" in rec:
                post_ids.append(str(rec["post_id"]))
            else:
                post_ids.append(str(rec))
        if post_ids:
            return post_ids

    # Fallback to newline separated IDs
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines

    raise ValueError(f"Unable to parse post IDs from {path}")


def _filter_retrieval(
    retrieval_path: Path,
    post_ids: Iterable[str],
    cids: Sequence[str] | None,
    top_k: int,
    tie_breaker: str,
) -> List[dict]:
    allowed_posts = set(post_ids)
    allowed_cids = set(cids) if cids else None
    filtered: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}

    for record in read_jsonl(retrieval_path):
        post_id = str(record.get("post_id"))
        cid = str(record.get("cid"))
        if post_id not in allowed_posts:
            continue
        if allowed_cids and cid not in allowed_cids:
            continue

        candidates = [
            _normalize_candidate(entry) for entry in record.get("candidates", [])
        ]
        ordered = _sort_candidates(candidates, tie_breaker)
        filtered[(post_id, cid)] = ordered[:top_k]

    output = []
    for (post_id, cid), candidates in sorted(filtered.items()):
        output.append(
            {
                "post_id": post_id,
                "cid": cid,
                "candidates": [
                    {"sent_id": sent_id, "score": score} for sent_id, score in candidates
                ],
            }
        )
    return output


def main(
    runtime_cfg: str,
    retrieval_jsonl: str,
    posts_jsonl: str,
    cids: Sequence[str] | None,
    k: int | None,
    out_path: str,
) -> None:
    cfg = load_config(runtime_cfg)
    scope = _cfg_get(cfg, "retrieval.scope", "same_post")
    if scope != "same_post":
        raise ValueError(
            f"Runtime config must enforce same_post scope, got {scope!r}"
        )
    tie_breaker = _cfg_get(cfg, "retrieval.tie_breaker", "score_then_id")
    default_k = _cfg_get(cfg, "retrieval.k_infer_default", 20)
    top_k = k or default_k
    min_with_post_len = bool(_cfg_get(cfg, "retrieval.min_with_post_len", True))

    post_ids = _load_post_ids(Path(posts_jsonl))
    if not post_ids:
        logger.warning("No posts found in %s; nothing to do", posts_jsonl)
        return

    retrieval_path = Path(retrieval_jsonl)
    if not retrieval_path.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_jsonl}")

    records = _filter_retrieval(
        retrieval_path=retrieval_path,
        post_ids=post_ids,
        cids=cids,
        top_k=top_k,
        tie_breaker=tie_breaker,
    )

    if min_with_post_len:
        # Candidates list is already capped by available retrieval entries,
        # so the condition is implicitly satisfied.
        logger.debug("min_with_post_len enforced via retrieval cap")

    write_jsonl(Path(out_path), records)
    logger.info(
        "Wrote %d retrieval entries (k=%d) to %s",
        len(records),
        top_k,
        out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fold-aware retrieval filtering CLI")
    parser.add_argument("--runtime_cfg", required=True, help="Path to runtime YAML")
    parser.add_argument(
        "--retrieval_jsonl",
        required=True,
        help="Path to full retrieval JSONL (same-post candidates)",
    )
    parser.add_argument(
        "--posts_jsonl",
        required=True,
        help="JSON/JSONL file containing post_id entries for the target fold",
    )
    parser.add_argument("--cids", nargs="+", help="Criterion IDs to include (optional)")
    parser.add_argument(
        "--k",
        type=int,
        help="Top-K candidates to keep (defaults to runtime k_infer_default)",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    args = parser.parse_args()

    main(
        runtime_cfg=args.runtime_cfg,
        retrieval_jsonl=args.retrieval_jsonl,
        posts_jsonl=args.posts_jsonl,
        cids=args.cids,
        k=args.k,
        out_path=args.out,
    )

