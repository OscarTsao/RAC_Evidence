"""Run Optuna HPO for cross-encoders with GPU-parallel workers."""

from __future__ import annotations

import argparse

from Project.hpo.crossenc import run_hpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-encoder HPO with Optuna.")
    parser.add_argument("--cfg", default="configs/ce_sc.yaml", help="Config path.")
    parser.add_argument("--task", choices=["sc", "pc"], default="sc", help="Task to optimize.")
    parser.add_argument("--n_trials", type=int, help="Total Optuna trials (defaults to config).")
    parser.add_argument("--n_jobs", type=int, help="Parallel workers (defaults to GPU count).")
    parser.add_argument("--study_name", help="Optuna study name.")
    parser.add_argument("--storage", help="Optuna storage URI.")
    args = parser.parse_args()
    run_hpo(
        cfg_path=args.cfg,
        task=args.task,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
