"""Shared helpers for graph utilities."""

from __future__ import annotations

from typing import Any

import torch

try:  # pragma: no cover - optional dependency
    from torch_geometric.data import HeteroData  # type: ignore

    HAS_PYG = True
except Exception:  # pragma: no cover
    HAS_PYG = False

    class _Store(dict):
        def __getattr__(self, item: str) -> Any:
            return self[item]

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    class HeteroData(dict):
        """Very small shim mimicking torch_geometric.data.HeteroData."""

        def __getitem__(self, key: Any) -> _Store:
            if key not in self:
                super().__setitem__(key, _Store())
            return super().__getitem__(key)

        def __setitem__(self, key: Any, value: Any) -> None:
            super().__setitem__(key, value)

        def to(self, device: torch.device) -> "HeteroData":
            for store in self.values():
                if isinstance(store, dict):
                    for k, v in store.items():
                        if torch.is_tensor(v):
                            store[k] = v.to(device)
            return self
