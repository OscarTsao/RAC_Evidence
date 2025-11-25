from pathlib import Path

import torch

from Project.graph.build_graph import build_graphs
from Project.graph.hetero_gnn import SimpleHeteroGNN
from Project.utils.hydra_utils import load_config


def _ensure_graphs() -> Path:
    graphs_path = Path("data/processed/graphs.pt")
    if not graphs_path.exists():
        # create minimal CE outputs if missing
        interim = Path("data/interim")
        interim.mkdir(parents=True, exist_ok=True)
        from Project.utils.data import load_raw_dataset
        import json as _json

        dataset = load_raw_dataset(Path("data/raw"))
        if not (interim / "ce_sc_scores.jsonl").exists():
            sc_records = []
            for lab in dataset["labels_sc"]:
                sc_records.append(
                    {
                        "post_id": lab.post_id,
                        "cid": lab.cid,
                        "sent_id": lab.sent_id,
                        "logit": 1.0 if lab.label == 1 else -0.5,
                        "prob": 0.7 if lab.label == 1 else 0.3,
                    }
                )
            (interim / "ce_sc_scores.jsonl").write_text("\n".join(_json.dumps(r) for r in sc_records))
        if not (interim / "ce_pc_post.jsonl").exists():
            pc_records = []
            for lab in dataset["labels_pc"]:
                pc_records.append(
                    {
                        "post_id": lab.post_id,
                        "cid": lab.cid,
                        "agg": "max",
                        "logit": 1.0 if lab.label == 1 else -0.4,
                        "prob": 0.7 if lab.label == 1 else 0.3,
                    }
                )
            (interim / "ce_pc_post.jsonl").write_text("\n".join(_json.dumps(r) for r in pc_records))
        cfg = load_config("configs/graph.yaml")
        build_graphs(cfg)
    return graphs_path


def test_gnn_forward_shapes() -> None:
    graphs_path = _ensure_graphs()
    graphs = torch.load(graphs_path, weights_only=False)
    graph = graphs[0]
    model = SimpleHeteroGNN()
    edge_logits, node_logits = model(graph)
    if ("sentence", "supports", "criterion") in graph:
        num_edges = graph[("sentence", "supports", "criterion")]["edge_attr"].shape[0]
        assert edge_logits.shape[0] == num_edges
    assert node_logits.shape[0] == graph["criterion"]["x"].shape[0]
