"""Select per-criterion K from sweep metrics based on recall/coverage gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf

from Project.retrieval.utils import write_json
from Project.utils.hydra_utils import cfg_get as _cfg_get, load_config
from Project.utils.logging import get_logger


def _load_results(sweep_json: Path) -> List[dict]:
    data = json.loads(sweep_json.read_text())
    return data.get("results", [])


def _metrics_for_k(entry: dict, cid: str | None = None) -> Tuple[float, float]:
    if cid is None:
        block = entry.get("overall", {})
    else:
        block = entry.get("per_criterion", {}).get(cid, {})
    return float(block.get("recall_at_k", 0.0)), float(block.get("coverage_at_k", 0.0))


def _pick_smallest_meeting(
    entries: List[dict],
    cid: str | None,
    recall_gate: float,
    coverage_gate: float,
) -> int | None:
    for entry in entries:
        recall, cov = _metrics_for_k(entry, cid)
        if recall >= recall_gate and cov >= coverage_gate:
            return int(entry["k"])
    return None


def _best_recall_k(entries: List[dict], cid: str | None) -> int:
    best_k = int(entries[0]["k"])
    best_recall = -1.0
    for entry in entries:
        recall, _ = _metrics_for_k(entry, cid)
        if recall > best_recall or (recall == best_recall and entry["k"] < best_k):
            best_recall = recall
            best_k = int(entry["k"])
    return best_k


def select_k(
    sweep_json: Path,
    gates: DictConfig,
    dynamic_policy: DictConfig,
    prefer_min_k: bool = True,
) -> dict:
    logger = get_logger(__name__)
    entries = sorted(_load_results(sweep_json), key=lambda x: x["k"])
    if not entries:
        raise ValueError(f"No sweep results found at {sweep_json}")

    # Identify criteria present
    criteria_ids = set()
    for entry in entries:
        criteria_ids.update(entry.get("per_criterion", {}).keys())

    per_criterion_choice: Dict[str, int] = {}
    needs_help: List[str] = []
    recall_gate = float(_cfg_get(gates, "recall_per_criterion_min", 0.0))
    coverage_gate = float(_cfg_get(gates, "coverage_per_criterion_min", 0.0))
    max_k = int(_cfg_get(dynamic_policy, "max_k", entries[-1]["k"]))

    for cid in criteria_ids:
        smallest = _pick_smallest_meeting(entries, cid, recall_gate, coverage_gate)
        if smallest is None:
            smallest = _best_recall_k(entries, cid)
        per_criterion_choice[cid] = smallest
        recall, cov = _metrics_for_k(next(e for e in entries if e["k"] == smallest), cid)
        if recall < recall_gate or cov < coverage_gate:
            needs_help.append(cid)

    overall_recall_gate = float(_cfg_get(gates, "recall_overall_min", 0.0))
    overall_k = _pick_smallest_meeting(entries, None, overall_recall_gate, 0.0)
    if overall_k is None:
        overall_k = _best_recall_k(entries, None)

    global_default_k_cfg = int(_cfg_get(dynamic_policy, "global_default_k", overall_k))
    available_ks = {int(e["k"]) for e in entries}
    if prefer_min_k:
        global_default_k = overall_k
    else:
        global_default_k = global_default_k_cfg if global_default_k_cfg in available_ks else overall_k

    k_overrides: Dict[str, int] = {}
    sensitive = set(_cfg_get(dynamic_policy, "sensitive_criteria", []))
    sensitive_cap = int(_cfg_get(dynamic_policy, "sensitive_cap_k", max_k))

    for cid, chosen_k in per_criterion_choice.items():
        recall_default, cov_default = _metrics_for_k(
            next(e for e in entries if e["k"] == global_default_k),
            cid,
        )
        if recall_default >= recall_gate and cov_default >= coverage_gate:
            continue
        candidate_k = chosen_k
        if cid in sensitive:
            # find smallest within cap meeting gates
            within_cap = [e for e in entries if int(e["k"]) <= sensitive_cap]
            cap_k = _pick_smallest_meeting(within_cap, cid, recall_gate, coverage_gate)
            if cap_k is not None:
                candidate_k = cap_k
        k_overrides[cid] = int(candidate_k)
        recall_cand, cov_cand = _metrics_for_k(next(e for e in entries if e["k"] == candidate_k), cid)
        if recall_cand < recall_gate or cov_cand < coverage_gate:
            if cid not in needs_help:
                needs_help.append(cid)

    selection = {
        "k_infer_default": int(global_default_k if global_default_k else overall_k),
        "k_infer_overrides": k_overrides,
        "overall_gate_k": int(overall_k),
        "needs_help": sorted(set(needs_help)),
    }
    logger.info(
        "Selected K default=%s overrides=%s needs_help=%s",
        selection["k_infer_default"],
        selection["k_infer_overrides"],
        selection["needs_help"],
    )
    return selection


def write_selection_artifacts(
    selection: dict,
    sweep_cfg: DictConfig,
    exp: str,
) -> Path:
    out_dir = Path(str(_cfg_get(sweep_cfg, "artifacts.out_dir", "outputs/runs/${EXP}/retrieval_sweep")).replace("${EXP}", exp))
    out_path = out_dir / str(_cfg_get(sweep_cfg, "artifacts.selected_k_json", "selected_k.json"))
    write_json(out_path, selection)
    return out_path


def update_runtime_config(runtime_path: Path, selection: dict) -> None:
    cfg = OmegaConf.load(str(runtime_path))
    if "retrieval" not in cfg:
        cfg.retrieval = {}
    cfg.retrieval.k_infer_default = selection["k_infer_default"]
    cfg.retrieval.k_infer_overrides = selection["k_infer_overrides"]
    OmegaConf.save(cfg, runtime_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_json", required=True)
    parser.add_argument("--cfg", default="configs/retrieval/k_selection.yaml")
    parser.add_argument("--runtime_out", default="configs/retrieval/runtime.yaml")
    parser.add_argument("--exp", default=None)
    parser.add_argument("--prefer_min_k", action="store_true", help="Prefer smallest K that meets gates")
    args = parser.parse_args()

    sweep_cfg = load_config(args.cfg)
    exp = args.exp or sweep_cfg.get("exp", "demo")
    sweep_cfg.EXP = exp
    selection = select_k(
        Path(args.sweep_json),
        gates=sweep_cfg.gates,
        dynamic_policy=sweep_cfg.dynamic_policy,
        prefer_min_k=bool(args.prefer_min_k),
    )
    write_selection_artifacts(selection, sweep_cfg, exp)
    update_runtime_config(Path(args.runtime_out), selection)


if __name__ == "__main__":
    main()
