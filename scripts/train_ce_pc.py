"""Train BGE reranker as CE-PC on the real dataset (5-fold)."""

from __future__ import annotations

import argparse

from Project.crossenc.hf_trainer import train_ce


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ce_pc_bge.yaml")
    args = parser.parse_args()
    train_ce(args.cfg, task="pc")


if __name__ == "__main__":
    main()
