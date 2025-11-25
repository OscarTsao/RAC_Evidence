"""Rich logging helper."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name if name else "Project")
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
