"""Optuna HPO runner for cross-encoders with GPU-aware parallelism."""

from __future__ import annotations

import copy
import math
import os
import random
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from transformers import set_seed

from Project.crossenc.hf_trainer import _build_pc_examples, _build_sc_examples, _train_fold
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger
from Project.utils.optuna_utils import create_study

logger = get_logger(__name__)


def _maybe_set_gpu(gpu_id: int | None) -> None:
    if gpu_id is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def _sample_hparams(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    params = {
        "train.lr": trial.suggest_float("train.lr", 5e-6, 5e-5, log=True),
        "train.batch_size": trial.suggest_categorical("train.batch_size", [8, 16, 32]),
        "train.gradient_accumulation": trial.suggest_categorical(
            "train.gradient_accumulation",
            [1, 2, 4],
        ),
        "train.warmup_ratio": trial.suggest_float("train.warmup_ratio", 0.0, 0.2),
        "train.weight_decay": trial.suggest_float("train.weight_decay", 0.0, 0.1),
        "train.epochs": trial.suggest_int("train.epochs", 2, 6),
        "train.grad_clip": trial.suggest_float("train.grad_clip", 0.5, 2.0),
    }
    # Cap batch size when running on CPU to avoid OOM
    if not torch.cuda.is_available() and params["train.batch_size"] > 16:
        params["train.batch_size"] = 16
    for key, value in params.items():
        OmegaConf.update(cfg, key, value, merge=True)
    return params


def _objective(
    trial: optuna.Trial,
    base_cfg: DictConfig,
    task: str,
    dataset: Dict[str, Iterable],
    output_root: Path,
) -> float:
    cfg = copy.deepcopy(base_cfg)
    _sample_hparams(trial, cfg)
    seed = int(_cfg_get(cfg, "seed", 42)) + int(trial.number)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if task == "sc":
        examples = _build_sc_examples(
            dataset["sentences"],
            dataset["criteria"],
            dataset["labels_sc"],
            neg_per_pos=int(_cfg_get(cfg, "data.neg_per_pos", 4)),
            seed=seed,
        )
    else:
        examples = _build_pc_examples(dataset["posts"], dataset["criteria"], dataset["labels_pc"])
    texts, labels = zip(*examples)
    trial_dir = output_root / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    score = _train_fold(list(texts), list(labels), cfg, trial_dir, fold=trial.number % 5)
    trial.set_user_attr("output_dir", str(trial_dir))
    torch.cuda.empty_cache()
    return score


def _run_worker(
    worker_id: int,
    cfg_path: str,
    task: str,
    study_name: str,
    storage: str,
    n_trials: int,
    gpu_id: int | None,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _maybe_set_gpu(gpu_id)
    cfg = load_config(cfg_path)
    base_cfg = copy.deepcopy(cfg)
    if not getattr(base_cfg, "exp", None):
        base_cfg.exp = "hpo"
    output_root = Path(
        _cfg_get(
            base_cfg,
            "output_dir",
            f"outputs/runs/{base_cfg.get('exp','hpo')}/{task}_hpo",
        )
    )
    dataset = load_raw_dataset(Path(base_cfg.data.raw_dir))
    study = create_study(
        study_name=study_name,
        storage_uri=storage,
        direction="maximize",
    )
    callbacks = [
        MaxTrialsCallback(
            n_trials,
            states=(TrialState.COMPLETE, TrialState.PRUNED),
        )
    ]
    objective = partial(
        _objective,
        base_cfg=base_cfg,
        task=task,
        dataset=dataset,
        output_root=output_root,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        callbacks=callbacks,
        gc_after_trial=True,
    )


def _normalize_storage(storage: str) -> str:
    if storage.startswith("sqlite:///"):
        db_path = storage.replace("sqlite:///", "", 1)
        storage = f"sqlite:///{Path(db_path).resolve()}"
    return storage


def run_hpo(
    cfg_path: str,
    task: str = "sc",
    n_trials: int | None = None,
    n_jobs: int | None = None,
    study_name: str | None = None,
    storage: str | None = None,
) -> optuna.study.Study:
    cfg = load_config(cfg_path)
    n_trials = n_trials or int(_cfg_get(cfg, "hpo_n_trials", 20))
    storage = _normalize_storage(storage or _cfg_get(cfg, "optuna_storage", "sqlite:///optuna.db"))
    if study_name is None:
        study_name = f"{cfg.get('exp','hpo')}_{task}"

    gpu_ids: List[int] = list(range(torch.cuda.device_count()))
    if n_jobs is None:
        n_jobs = max(1, len(gpu_ids)) or 1
    n_jobs = min(n_jobs, n_trials)
    if gpu_ids:
        n_jobs = min(n_jobs, len(gpu_ids))
    if not gpu_ids:
        logger.warning("No GPUs detected; running HPO on CPU with a single worker.")
        n_jobs = 1

    worker_trials = math.ceil(n_trials / max(1, n_jobs))
    ctx = get_context("spawn")
    procs = []
    for worker_id in range(n_jobs):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)] if gpu_ids else None
        p = ctx.Process(
            target=_run_worker,
            args=(worker_id, cfg_path, task, study_name, storage, worker_trials, gpu_id),
            daemon=False,
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            logger.error("HPO worker %s exited with code %s", p.name, p.exitcode)
    # Reload study to return the best result
    study = create_study(study_name=study_name, storage_uri=storage, direction="maximize")
    if study.best_trial:
        logger.info(
            "Best trial %s: value=%.4f params=%s",
            study.best_trial.number,
            study.best_value,
            study.best_params,
        )
    return study


__all__ = ["run_hpo"]
