"""Wrapper to build hetero graphs."""

from __future__ import annotations

from Project.graph.build_graph import main as build_main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    build_main(args.cfg)
