"""PyG Dataset for memory-efficient graph loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Dataset, HeteroData


class HeteroGraphDataset(Dataset):
    """Memory-efficient dataset for heterogeneous graphs.

    Loads graphs from individual .pt files on demand instead of loading all into memory.
    """

    def __init__(self, root: Path, transform=None, pre_transform=None):
        """Initialize dataset.

        Args:
            root: Directory containing data_{i}.pt files and metadata.json
            transform: Optional transform to apply on-the-fly
            pre_transform: Optional transform to apply during processing
        """
        self.root = Path(root)

        # Load metadata BEFORE calling parent __init__ to set _num_graphs
        metadata_path = self.root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self._num_graphs = self.metadata.get("num_graphs", 0)
        else:
            # Fallback: count files
            self._num_graphs = len(list(self.root.glob("data_*.pt")))
            self.metadata = {"num_graphs": self._num_graphs}

        super().__init__(str(root), transform, pre_transform)

    @property
    def raw_file_names(self):
        """Return list of raw file names."""
        return [f"data_{i}.pt" for i in range(self._num_graphs)]

    @property
    def processed_file_names(self):
        """Return list of processed file names."""
        # Graphs are already processed
        return self.raw_file_names

    def len(self):
        """Return number of graphs."""
        return self._num_graphs

    def get(self, idx: int) -> HeteroData:
        """Load a single graph by index.

        Args:
            idx: Graph index

        Returns:
            HeteroData graph
        """
        # Ensure root is a Path object
        root_path = Path(self.root) if not isinstance(self.root, Path) else self.root
        graph_path = root_path / f"data_{idx}.pt"
        if not graph_path.exists():
            raise IndexError(f"Graph {idx} not found at {graph_path}")

        data = torch.load(graph_path)
        return data

    def process(self):
        """No processing needed - graphs are already built."""
        pass


def load_hetero_dataset(graphs_dir: Path) -> HeteroGraphDataset:
    """Load heterogeneous graph dataset.

    Args:
        graphs_dir: Directory containing graph files

    Returns:
        HeteroGraphDataset instance
    """
    return HeteroGraphDataset(graphs_dir)
