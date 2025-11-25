# AI/ML Experiment Template

Minimal template for ML experiments using PyTorch, Transformers, MLflow, and Optuna.

## Quickstart

- Python 3.10+ recommended.
- Create and activate a virtual environment, then install:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Layout

- `src/Project/models/model.py` – example model wrapper for Transformers.
- `src/Project/utils/` – utility helpers (`get_logger`, `set_seed`, MLflow helpers).
- `src/Project/engine/` – training and evaluation loops.
- `src/Project/data/` – lightweight dataset helpers.
- `mlruns/` – canonical location for MLflow tracking data.
- `optuna.db` – SQLite database used by Optuna for HPO studies.
- `outputs/` – suggested place for artifacts.

## MLflow

Configure MLflow for local tracking and run logging:

```python
from Project.utils import configure_mlflow, enable_autologging, mlflow_run

# MLflow tracking data always lives in ./mlruns
configure_mlflow(tracking_uri="file:./mlruns", experiment="demo")
enable_autologging()

with mlflow_run("hello", tags={"stage": "dev"}, params={"lr": 1e-4}):
    # your training loop here
    pass
```

## Optuna

Optuna studies are stored in `optuna.db` so completed trials persist locally:

```python
import optuna

STORAGE_URI = "sqlite:///optuna.db"  # always points at repo-level optuna.db

study = optuna.create_study(
    study_name="demo",
    direction="maximize",
    storage=STORAGE_URI,
    load_if_exists=True,
)
```

## Development

- Run linters/formatters:
```
ruff check src tests
black src tests
```
- Run tests (add your own under `tests/`):
```
pytest
```

