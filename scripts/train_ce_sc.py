"""Wrapper to train CE-SC."""

from __future__ import annotations

from Project.crossenc.train_ce_sc import main as train_main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    train_main(args.cfg)
