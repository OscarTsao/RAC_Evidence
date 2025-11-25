"""Shared dataset loading helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from Project.dataio.schemas import Criterion, LabelPC, LabelSC, Post, Sentence
from Project.utils.io import load_models


def default_tokenize(text: str) -> list[str]:
    """Simple word tokenizer using regex.

    Args:
        text: Input text

    Returns:
        List of lowercase word tokens

    Examples:
        >>> default_tokenize("Hello, World!")
        ['hello', 'world']
        >>> default_tokenize("PyTorch 2.0 is great!")
        ['pytorch', '2', '0', 'is', 'great']
    """
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]


def load_raw_dataset(raw_dir: Path) -> Dict[str, List[object]]:
    """Load raw JSONL files and return typed lists."""
    posts = load_models(raw_dir / "posts.jsonl", Post)
    sentences = load_models(raw_dir / "sentences.jsonl", Sentence)
    labels_sc = load_models(raw_dir / "labels_sc.jsonl", LabelSC)
    labels_pc = load_models(raw_dir / "labels_pc.jsonl", LabelPC)
    criteria_data = json.loads((raw_dir / "criteria.json").read_text())
    criteria = [Criterion.model_validate(c) for c in criteria_data]
    return {
        "posts": posts,
        "sentences": sentences,
        "labels_sc": labels_sc,
        "labels_pc": labels_pc,
        "criteria": criteria,
    }
