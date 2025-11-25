"""Wrapper to run CE-PC inference."""

from __future__ import annotations

from Project.crossenc.infer_ce_pc import main as infer_main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    infer_main(args.cfg)
