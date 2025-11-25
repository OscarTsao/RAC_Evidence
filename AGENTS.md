# Repository Guidelines

Reference checklist for contributing to this template-based ML experimentation repo. Keep changes small, tested, and reproducible.

## Project Structure & Modules
- Core code lives in `src/Project/`: `models/` (model wrappers), `engine/` (train/eval loops), `utils/` (logging, seeding, MLflow helpers).
- Scripts for dataset generation sit in `scripts/` with a companion `scripts/README.md`.
- Data inputs and generated groundtruths live under `data/` (`redsm5/`, `groundtruth/`, `DSM5/`); experiment artifacts default to `outputs/` and MLflow runs to `mlruns/`.
- Optuna HPO storage is standardized on the repo-level `optuna.db` (use `sqlite:///optuna.db`).
- Tests belong in `tests/` (currently sparse); docs in `docs/`; configs and example runs under `configs/` and `Optimization_*` folders.

## Build, Test, and Development Commands
- Create env and install: `python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'`.
- Lint: `ruff check src tests` (line length 100); format: `black src tests`.
- Type-check (optional but encouraged): `mypy src tests`.
- Run tests: `pytest` (uses `-q` from `pyproject.toml`); add `-s` when debugging.
- Regenerate datasets when annotations change: run the three `python3 scripts/generate_*` commands (see `scripts/README.md` for outputs and schemas).

## Coding Style & Naming Conventions
- Target Python 3.10+; prefer type hints on public functions.
- Follow Black formatting; keep imports sorted by Ruff’s defaults.
- Use `snake_case` for functions/variables, `PascalCase` for classes, UPPER_SNAKE for constants; keep module names lowercase.
- Logging: use helpers in `Project.utils.log` instead of `print`; seed experiments via `Project.utils.seed.set_seed`.

## Testing Guidelines
- Place unit/integration tests in `tests/` mirroring source structure; name files `test_*.py` and functions `test_*`.
- Use realistic fixtures from `data/redsm5/` when possible and keep temporary artifacts in `outputs/` (clean up after assertions).
- Add coverage for new behaviors, especially around data loading, training loops, and MLflow logging paths.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject lines (e.g., “Add eval loop retries”); group related changes and include rationale in the body when non-obvious.
- PRs: summarize intent, list key changes and validation (lint/tests/regen scripts), and note data or config updates; link to issues/tasks when applicable and include screenshots or logs for user-facing or metrics changes.
