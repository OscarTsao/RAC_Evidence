"""Dataset helpers, chunkers, and collators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

try:
    from transformers import PreTrainedTokenizerBase
except Exception:  # pragma: no cover - transformers might be missing
    PreTrainedTokenizerBase = Any  # type: ignore

from Project.dataio.schemas import (
    Criterion,
    LabelPC,
    LabelSC,
    Post,
    Sentence,
)


class SimpleTokenizer:
    """Lightweight tokenizer that mimics transformers behaviour for tests."""

    def __init__(self, vocab_size: int = 50257) -> None:
        self.vocab_size = vocab_size
        # reserve 0 for padding, 1 for CLS, 2 for SEP
        self.cls_id = 1
        self.sep_id = 2

    def _encode(self, text: str) -> List[int]:
        tokens = text.split()
        return [3 + (abs(hash(tok)) % (self.vocab_size - 3)) for tok in tokens]

    def __call__(
        self,
        text: str,
        text_pair: Optional[str] = None,
        truncation: Optional[str] = None,
        max_length: Optional[int] = None,
        padding: str | bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        tokens_first = self._encode(text)
        tokens_second: List[int] = self._encode(text_pair) if text_pair else []
        if truncation == "only_first" and max_length:
            max_first = max_length - len(tokens_second) - 3
            max_first = max(max_first, 0)
            tokens_first = tokens_first[:max_first]
        input_ids = [self.cls_id] + tokens_first + [self.sep_id]
        token_type_ids = [0] * len(input_ids)
        if tokens_second:
            input_ids += tokens_second + [self.sep_id]
            token_type_ids += [1] * (len(tokens_second) + 1)
        attention_mask = [1] * len(input_ids)
        if padding and max_length:
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [0] * pad_len
                attention_mask += [0] * pad_len
                token_type_ids += [0] * pad_len
            else:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                token_type_ids = token_type_ids[:max_length]
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        if return_tensors == "pt":
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        return result


def resolve_tokenizer(tokenizer: Optional[PreTrainedTokenizerBase] = None) -> Any:
    """Return a user-supplied tokenizer or fall back to SimpleTokenizer."""
    if tokenizer is not None:
        return tokenizer
    try:  # pragma: no cover - best-effort real tokenizer
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-bert", local_files_only=True
        )
    except Exception:
        return SimpleTokenizer()


def sentence_chunker(text: str, max_len: int = 512, stride: int = 128) -> List[Dict[str, Any]]:
    """Sliding-window chunker for long text."""
    tokens = text.split()
    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + max_len, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append({"chunk_id": f"chunk_{idx}", "text": chunk_text, "start": start, "end": end})
        if end == len(tokens):
            break
        start = end - stride
        idx += 1
    if not chunks:  # pragma: no cover - defensive
        chunks.append({"chunk_id": "chunk_0", "text": "", "start": 0, "end": 0})
    return chunks


@dataclass
class PairExample:
    text: str
    criterion: str
    label: int
    meta: Dict[str, Any]


class PairDataset(Dataset[PairExample]):
    """Dataset of text/criterion pairs."""

    def __init__(self, examples: Sequence[PairExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PairExample:
        return self.examples[idx]


def collate_pairwise_ce(
    batch: Sequence[PairExample],
    tokenizer: Optional[Any] = None,
    max_length: int = 256,
    truncation: str = "only_first",
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of PairExample with truncation applied to the TEXT side only."""
    tk = resolve_tokenizer(tokenizer)
    encoded = [
        tk(
            example.text,
            example.criterion,
            truncation=truncation,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        for example in batch
    ]
    input_ids = torch.stack([e["input_ids"] for e in encoded], dim=0)
    attention_mask = torch.stack([e["attention_mask"] for e in encoded], dim=0)
    token_type_ids = torch.stack([e["token_type_ids"] for e in encoded], dim=0)
    labels = torch.tensor([ex.label for ex in batch], dtype=torch.float)
    meta = [ex.meta for ex in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
        "meta": meta,
    }


def attach_post_lookup(
    posts: Iterable[Post],
    sentences: Iterable[Sentence],
) -> Dict[str, Dict[str, Sentence]]:
    """Build a mapping post_id -> sent_id -> sentence for quick lookup."""
    post_map: Dict[str, Dict[str, Sentence]] = {}
    for sent in sentences:
        post_map.setdefault(sent.post_id, {})[sent.sent_id] = sent
    # ensure every post id is present
    for post in posts:
        post_map.setdefault(post.post_id, {})
    return post_map


def load_basic_collections(
    posts: Sequence[Post],
    sentences: Sequence[Sentence],
    criteria: Sequence[Criterion],
    labels_sc: Sequence[LabelSC],
    labels_pc: Sequence[LabelPC],
) -> Dict[str, Any]:
    """Create convenience lookup dictionaries used across modules."""
    sent_lookup = attach_post_lookup(posts, sentences)
    sc_lookup: Dict[Tuple[str, str], int] = {}
    for lab in labels_sc:
        sc_lookup[(lab.post_id, lab.sent_id, lab.cid)] = lab.label
    pc_lookup: Dict[Tuple[str, str], int] = {(lab.post_id, lab.cid): lab.label for lab in labels_pc}
    criterion_map = {c.cid: c for c in criteria}
    return {
        "sent_lookup": sent_lookup,
        "sc_lookup": sc_lookup,
        "pc_lookup": pc_lookup,
        "criterion_map": criterion_map,
    }
