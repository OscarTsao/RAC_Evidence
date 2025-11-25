"""Lightweight I/O helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Project root for path validation
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()


def _validate_output_path(path: Path) -> Path:
    """Validate that output path is within project directory.

    Args:
        path: Path to validate

    Returns:
        Resolved path

    Raises:
        ValueError: If path is outside project directory
    """
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(
            f"Path {path} resolves to {resolved} which is outside "
            f"project directory {PROJECT_ROOT}. This is not allowed for security reasons."
        )
    return resolved


def read_jsonl(path: Path) -> List[dict]:
    """Read JSONL file with error handling for corrupted lines."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(data)
            except json.JSONDecodeError as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Skipping invalid JSON on line {i} in {path}: {e}")
                continue
    return items


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Write records to JSONL file with path validation.

    Args:
        path: Output path (must be within project directory)
        records: Iterable of dictionaries to write

    Raises:
        ValueError: If path is outside project directory
    """
    validated_path = _validate_output_path(path)
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    with validated_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_models(path: Path, model: Type[T]) -> List[T]:
    return [model.model_validate(rec) for rec in read_jsonl(path)]


def dump_models(path: Path, data: Sequence[BaseModel]) -> None:
    write_jsonl(path, [d.model_dump() for d in data])
