"""Train PC cross-encoder v2 with strict CV and calibrated OOF outputs."""

from __future__ import annotations

import argparse

from Project.crossenc.train_pc_v2 import run_training
from Project.utils.hydra_utils import load_config
from Project.utils.logging import get_logger


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PC v2 cross-encoder with strict CV.")
    parser.add_argument("--cfg", default="configs/ce_pc_v2.yaml", help="Config path")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    result_paths = run_training(cfg)
    logger.info("Completed PC v2 training. Artifacts: %s", result_paths)


if __name__ == "__main__":
    main()
