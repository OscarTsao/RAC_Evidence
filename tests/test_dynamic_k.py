import json
from pathlib import Path

from omegaconf import OmegaConf

from Project.retrieval.dynamic_k import select_k


def _mock_sweep(tmp_path: Path) -> Path:
    data = {
        "results": [
            {
                "k": 10,
                "overall": {"recall_at_k": 0.8, "coverage_at_k": 0.8, "p95_latency_ms": 10},
                "per_criterion": {
                    "c1": {"recall_at_k": 0.8, "coverage_at_k": 0.8},
                    "c2": {"recall_at_k": 0.7, "coverage_at_k": 0.7},
                },
            },
            {
                "k": 20,
                "overall": {"recall_at_k": 0.95, "coverage_at_k": 0.95, "p95_latency_ms": 12},
                "per_criterion": {
                    "c1": {"recall_at_k": 0.95, "coverage_at_k": 0.95},
                    "c2": {"recall_at_k": 0.92, "coverage_at_k": 0.9},
                },
            },
        ]
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(data))
    return path


def test_select_k_overrides(tmp_path: Path):
    sweep = _mock_sweep(tmp_path)
    gates = OmegaConf.create(
        {
            "recall_overall_min": 0.9,
            "recall_per_criterion_min": 0.9,
            "coverage_per_criterion_min": 0.9,
        }
    )
    policy = OmegaConf.create(
        {
            "global_default_k": 10,
            "sensitive_criteria": ["c2"],
            "sensitive_cap_k": 50,
            "max_k": 100,
        }
    )
    selection = select_k(sweep, gates, policy)
    assert selection["k_infer_default"] == 10  # global default acceptable overall
    assert selection["k_infer_overrides"]["c2"] == 20  # needs override to meet gate
