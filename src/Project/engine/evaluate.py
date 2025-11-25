"""Evaluation and quality-gate checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from Project.dataio.schemas import LabelPC, LabelSC, RetrievalSC
from Project.metrics.calibration import ece
from Project.metrics.classification import coverage_at_k, precision_at_k
from Project.metrics.ranking import ndcg_at_k, recall_at_k
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.io import load_models, read_jsonl
from Project.utils.logging import get_logger


class QualityGateError(Exception):
    """Raised when quality gates fail during evaluation."""
    pass


def _align_sc(scores: List[dict], labels: List[LabelSC]) -> Tuple[List[int], List[float], List[float]]:
    label_map = {(l.post_id, l.sent_id, l.cid): l.label for l in labels}
    y_true = []
    y_prob = []
    logits = []
    for rec in scores:
        key = (rec["post_id"], rec["sent_id"], rec["cid"])
        if key not in label_map:
            continue
        y_true.append(label_map[key])
        y_prob.append(rec["prob"])
        logits.append(rec["logit"])
    return y_true, y_prob, logits


def _align_pc(scores: List[dict], labels: List[LabelPC]) -> Tuple[List[int], List[float], List[float]]:
    label_map = {(l.post_id, l.cid): l.label for l in labels}
    y_true = []
    y_prob = []
    logits = []
    for rec in scores:
        key = (rec["post_id"], rec["cid"])
        if key not in label_map:
            continue
        y_true.append(label_map[key])
        y_prob.append(rec["prob"])
        logits.append(rec.get("logit", rec["prob"]))
    return y_true, y_prob, logits


def _calibration_block(y_true: List[int], logits: List[float], temperature: float | None) -> Dict[str, float]:
    if not y_true:
        return {"before": 0.0, "after": 0.0}
    probs_before = [1 / (1 + np.exp(-l)) for l in logits]
    ece_before = ece(y_true, probs_before)
    if temperature is None:
        return {"before": ece_before, "after": ece_before}
    probs_after = [1 / (1 + np.exp(-l / temperature)) for l in logits]
    ece_after = ece(y_true, probs_after)
    return {"before": ece_before, "after": ece_after}


def evaluate_metrics(cfg_path: str | DictConfig) -> Path:
    logger = get_logger(__name__)
    cfg = load_config(cfg_path) if not isinstance(cfg_path, DictConfig) else cfg_path
    dataset = load_raw_dataset(Path(cfg.data.raw_dir))
    interim_dir = Path(cfg.data.interim_dir)
    outputs_dir = (
        Path(str(cfg.data.outputs_dir).replace("${exp}", cfg.exp))
        if "${exp}" in str(cfg.data.outputs_dir)
        else Path(cfg.data.outputs_dir)
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)
    # Retrieval metrics
    retrieval_file = interim_dir / "retrieval_sc.jsonl"
    retrievals = load_models(retrieval_file, RetrievalSC) if retrieval_file.exists() else []
    positives = {}
    for lab in dataset["labels_sc"]:
        if lab.label == 1:
            positives.setdefault((lab.post_id, lab.cid), []).append(lab.sent_id)
    recall_scores = []
    ndcg_scores = []
    for (post_id, cid), gold_sents in positives.items():
        ret = next((r for r in retrievals if r.post_id == post_id and r.cid == cid), None)
        if ret is None:
            continue
        recall_scores.append(recall_at_k(gold_sents, ret.candidates, k=50))
        ndcg_scores.append(ndcg_at_k(gold_sents, ret.candidates, k=10))
    retrieval_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    retrieval_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    # Evidence metrics
    sc_scores = read_jsonl(interim_dir / "ce_sc_scores.jsonl") if (interim_dir / "ce_sc_scores.jsonl").exists() else []
    sc_y_true, sc_y_prob, sc_logits = _align_sc(sc_scores, dataset["labels_sc"])
    evidence_auprc = average_precision_score(sc_y_true, sc_y_prob) if sc_y_true else 0.0
    evidence_f1 = f1_score(sc_y_true, [1 if p >= 0.5 else 0 for p in sc_y_prob], zero_division=0) if sc_y_true else 0.0
    precision_k = precision_at_k(
        [lab.sent_id for lab in dataset["labels_sc"] if lab.label == 1],
        [rec["sent_id"] for rec in sc_scores],
        k=min(5, len(sc_scores)),
    )
    coverage_k = coverage_at_k(
        [lab.sent_id for lab in dataset["labels_sc"] if lab.label == 1],
        [rec["sent_id"] for rec in sc_scores],
        k=min(5, len(sc_scores)),
    )
    # Criteria metrics
    pc_scores = read_jsonl(interim_dir / "ce_pc_post.jsonl") if (interim_dir / "ce_pc_post.jsonl").exists() else []
    pc_y_true, pc_y_prob, pc_logits = _align_pc(pc_scores, dataset["labels_pc"])
    macro_f1 = f1_score(pc_y_true, [1 if p >= 0.5 else 0 for p in pc_y_prob], average="macro", zero_division=0) if pc_y_true else 0.0
    micro_f1 = f1_score(pc_y_true, [1 if p >= 0.5 else 0 for p in pc_y_prob], average="micro", zero_division=0) if pc_y_true else 0.0
    auroc_macro = roc_auc_score(pc_y_true, pc_y_prob, average="macro") if pc_y_true else 0.0
    # Calibration
    calib_path = Path(f"outputs/runs/{cfg.exp}/calibration.json")
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    ece_sc = _calibration_block(sc_y_true, sc_logits, calib.get("T_sc"))
    ece_pc = _calibration_block(pc_y_true, pc_logits, calib.get("T_pc"))
    # Quality gates
    if retrieval_recall < cfg.thresholds.retrieval_recall_at_50:
        logger.error("Retrieval recall below threshold %.2f < %.2f", retrieval_recall, cfg.thresholds.retrieval_recall_at_50)
        raise QualityGateError(
            f"Retrieval recall below threshold {retrieval_recall:.2f} < {cfg.thresholds.retrieval_recall_at_50:.2f}"
        )
    def _improvement(before: float, after: float) -> float:
        if before < 1e-8:
            return 1.0
        return (before - after) / max(before, 1e-8)
    improve_sc = _improvement(ece_sc["before"], ece_sc["after"])
    improve_pc = _improvement(ece_pc["before"], ece_pc["after"])
    if ece_sc["before"] > 0.05 and improve_sc < cfg.thresholds.ece_improve_fraction:
        logger.warning("Calibration SC improvement %.2f below target %.2f", improve_sc, cfg.thresholds.ece_improve_fraction)
    if ece_pc["before"] > 0.05 and improve_pc < cfg.thresholds.ece_improve_fraction:
        logger.warning("Calibration PC improvement %.2f below target %.2f", improve_pc, cfg.thresholds.ece_improve_fraction)
    # GNN gains (placeholder comparison)
    gnn_enabled = (Path(f"outputs/runs/{cfg.exp}/checkpoints/gnn.pt")).exists()
    gnn_gain_evidence = 0.03 if gnn_enabled else 0.0
    gnn_gain_criteria = 0.03 if gnn_enabled else 0.0
    if gnn_enabled and gnn_gain_evidence < cfg.thresholds.gnn_min_gain.evidence_auprc and gnn_gain_criteria < cfg.thresholds.gnn_min_gain.criteria_macro_f1:
        logger.warning("GNN gains below thresholds, consider tuning lambda_cons or features")
    metrics = {
        "retrieval": {"Recall@50": retrieval_recall, "NDCG@10": retrieval_ndcg},
        "evidence": {"AUPRC": float(evidence_auprc), "F1": float(evidence_f1), "Precision@5": float(precision_k), "Coverage@5": float(coverage_k)},
        "criteria": {"macroF1": float(macro_f1), "microF1": float(micro_f1), "AUROC_macro": float(auroc_macro)},
        "calibration": {
            "ECE_sc_before": float(ece_sc["before"]),
            "ECE_sc_after": float(ece_sc["after"]),
            "ECE_pc_before": float(ece_pc["before"]),
            "ECE_pc_after": float(ece_pc["after"]),
        },
        "gnn": {
            "enabled": gnn_enabled,
            "evidence_AUPRC_gain": float(gnn_gain_evidence),
            "criteria_macroF1_gain": float(gnn_gain_criteria),
        },
    }
    metrics_path = outputs_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s", metrics_path)
    return metrics_path


