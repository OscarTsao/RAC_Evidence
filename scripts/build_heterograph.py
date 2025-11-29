#!/usr/bin/env python3
"""Script to build heterogeneous graphs from OOF predictions."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.graph.build_hetero import main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build heterogeneous graphs")
    parser.add_argument("--cfg", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.cfg)
