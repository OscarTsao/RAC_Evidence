"""Prepare data splits and sanity checks."""

from __future__ import annotations

from Project.dataio.prepare import prepare_data_splits

def main(cfg_path: str) -> None:
    prepare_data_splits(cfg_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
