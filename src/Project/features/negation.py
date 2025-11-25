"""Optional negation/hedge cue extractor (placeholder)."""

from __future__ import annotations

from typing import List

NEGATION_CUES = {"not", "no", "never", "without", "denies"}
HEDGE_CUES = {"maybe", "perhaps", "possible", "appears", "seems"}


def negation_flags(text: str) -> List[int]:
    """Return binary flags for negation cues per token."""
    tokens = text.lower().split()
    return [1 if tok in NEGATION_CUES else 0 for tok in tokens]


def hedge_flags(text: str) -> List[int]:
    """Return binary flags for hedge cues per token."""
    tokens = text.lower().split()
    return [1 if tok in HEDGE_CUES else 0 for tok in tokens]
