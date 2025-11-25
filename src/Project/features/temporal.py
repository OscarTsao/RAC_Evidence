"""Temporal cue extractor (placeholder)."""

from __future__ import annotations

import re
from typing import Optional


DURATION_PATTERN = re.compile(r"(\d+)\s*(day|days|week|weeks|month|months)")


def approximate_duration_days(text: str) -> Optional[int]:
    """Extract a rough duration in days if present."""
    match = DURATION_PATTERN.search(text.lower())
    if not match:
        return None
    value, unit = match.groups()
    mult = 1
    if unit.startswith("week"):
        mult = 7
    elif unit.startswith("month"):
        mult = 30
    return int(value) * mult
