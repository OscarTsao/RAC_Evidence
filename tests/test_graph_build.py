import json
from pathlib import Path

import torch

from Project.graph.build_graph import build_graphs
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config


def test_graph_build_creates_edges() -> None:
    dataset = load_raw_dataset(Path("data/raw"))
    interim = Path("data/interim")
    interim.mkdir(parents=True, exist_ok=True)
    sc_records = []
    for lab in dataset["labels_sc"]:
        sc_records.append(
            {
                "post_id": lab.post_id,
                "cid": lab.cid,
                "sent_id": lab.sent_id,
                "logit": 1.5 if lab.label == 1 else -0.2,
                "prob": 0.8 if lab.label == 1 else 0.2,
            }
        )
    pc_records = []
    for lab in dataset["labels_pc"]:
        pc_records.append(
            {
                "post_id": lab.post_id,
                "cid": lab.cid,
                "agg": "max",
                "logit": 1.2 if lab.label == 1 else -0.3,
                "prob": 0.7 if lab.label == 1 else 0.3,
            }
        )
    (interim / "ce_sc_scores.jsonl").write_text("\n".join(json.dumps(r) for r in sc_records))
    (interim / "ce_pc_post.jsonl").write_text("\n".join(json.dumps(r) for r in pc_records))
    cfg = load_config("configs/graph.yaml")
    output = build_graphs(cfg)
    graphs = torch.load(output, weights_only=False)
    assert graphs, "No graphs were built"
    graph = graphs[0]
    assert ("sentence", "supports", "criterion") in graph
    edge_attr = graph[("sentence", "supports", "criterion")]["edge_attr"]
    assert edge_attr.shape[1] == 6
