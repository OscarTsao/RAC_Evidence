"""Temperature scaling for probability calibration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import torch
from omegaconf import DictConfig

from Project.metrics.calibration import ece, nll
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.io import read_jsonl
from Project.utils.logging import get_logger


class TemperatureScaler:
    """Temperature scaling wrapper for probability calibration."""

    def __init__(self, lr: float = 0.01, steps: int = 200):
        """Initialize temperature scaler.

        Args:
            lr: Learning rate for optimization
            steps: Number of optimization steps
        """
        self.lr = lr
        self.steps = steps
        self.temperature = 1.0

    def fit(self, logits: Iterable[float], labels: Iterable[int]) -> float:
        """Fit temperature parameter.

        Args:
            logits: Model logits
            labels: Ground truth labels

        Returns:
            Optimal temperature value
        """
        self.temperature = learn_temperature(logits, labels, lr=self.lr, steps=self.steps)
        return self.temperature


def learn_temperature(
    logits: Iterable[float],
    labels: Iterable[int],
    lr: float = 0.01,
    steps: int = 200,
) -> float:
    logit_tensor = torch.tensor(list(logits), dtype=torch.float32)
    label_tensor = torch.tensor(list(labels), dtype=torch.float32)
    temperature = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([temperature], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        scaled_logits = logit_tensor / torch.clamp(temperature, min=1e-3)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled_logits, label_tensor)
        loss.backward()
        optimizer.step()
    return float(torch.clamp(temperature, min=1e-3).item())


def calibrate_file(scores_path: Path, label_key: str = "label") -> Tuple[float, float, float]:
    """Calibrate a JSONL file that contains logit + label fields."""
    records = read_jsonl(scores_path)
    logits = [r["logit"] for r in records]
    labels = [r[label_key] for r in records]
    # Vectorized sigmoid computation
    logits_t = torch.tensor(logits, dtype=torch.float32)
    probs_before = torch.sigmoid(logits_t).tolist()
    before = {
        "nll": nll(labels, logits),
        "ece": ece(labels, probs_before),
    }
    temp = learn_temperature(logits, labels)
    scaled_probs = torch.sigmoid(logits_t / temp).tolist()
    after = {"nll": nll(labels, logits, temperature=temp), "ece": ece(labels, scaled_probs)}
    return temp, before["ece"], after["ece"]


def _calibrate_with_labels(records: list[dict], label_lookup: dict, label_keys: tuple[str, ...]) -> Tuple[float, float, float]:
    logits = []
    labels = []
    for rec in records:
        key = tuple(rec[k] for k in label_keys)
        label = label_lookup.get(key)
        if label is None:
            continue
        logits.append(rec["logit"])
        labels.append(label)
    return calibrate_values(logits, labels)


def calibrate_values(logits: Iterable[float], labels: Iterable[int]) -> Tuple[float, float, float]:
    # Vectorized sigmoid computation for better performance
    logits_t = torch.tensor(list(logits), dtype=torch.float32)
    probs_before = torch.sigmoid(logits_t).tolist()
    before_ece = ece(labels, probs_before)
    temp = learn_temperature(logits, labels)
    after_probs = torch.sigmoid(logits_t / temp).tolist()
    after_ece = ece(labels, after_probs)
    return temp, before_ece, after_ece


def run_calibration(cfg: DictConfig) -> Path:
    logger = get_logger(__name__)
    exp = cfg.get("exp", "debug")
    output_setting = cfg.get("output", f"outputs/runs/{exp}/calibration.json")
    output_path = Path(str(output_setting).replace("${exp}", exp))
    if output_path.suffix:  # explicit file path
        calib_out = output_path
        out_dir = output_path.parent
    else:
        out_dir = output_path
        calib_out = out_dir / "calibration.json"
    if calib_out.exists() and calib_out.is_dir():  # cleanup legacy dirs
        shutil.rmtree(calib_out)
    calib_out.parent.mkdir(parents=True, exist_ok=True)
    temps = {}
    dataset = load_raw_dataset(Path(cfg.data.raw_dir))
    sc_labels = {(l.post_id, l.sent_id, l.cid): l.label for l in dataset["labels_sc"]}
    pc_labels = {(l.post_id, l.cid): l.label for l in dataset["labels_pc"]}
    if "sc" in cfg.task:
        sc_records = read_jsonl(Path(cfg.inputs.sc))
        temp_sc, _, _ = _calibrate_with_labels(sc_records, sc_labels, ("post_id", "sent_id", "cid"))
        temps["T_sc"] = temp_sc
    if "pc" in cfg.task:
        pc_records = read_jsonl(Path(cfg.inputs.pc))
        temp_pc, _, _ = _calibrate_with_labels(pc_records, pc_labels, ("post_id", "cid"))
        temps["T_pc"] = temp_pc
    calib_out.write_text(json.dumps(temps, indent=2))
    logger.info("Saved calibration parameters to %s", calib_out)
    return calib_out


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    run_calibration(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
