"""Wrapper to run temperature scaling calibration."""

from __future__ import annotations

from Project.calib.temperature import main as calib_main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    calib_main(args.cfg)
