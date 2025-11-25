from pathlib import Path

from Project.metrics.ranking import recall_at_k
from Project.retrieval.index_faiss import build_per_post_index
from Project.retrieval.retrieve import retrieve_topk
from Project.retrieval.train_bi import train_bi_encoder
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config


def test_retrieval_recall_same_post() -> None:
    data = load_raw_dataset(Path("data/raw"))
    model = train_bi_encoder(data["sentences"], data["criteria"], Path("outputs/runs/test/bi"), seed=0)
    index = build_per_post_index(model, data["sentences"], Path("data/interim"), use_faiss=False)
    criteria_map = {c.cid: c for c in data["criteria"]}
    cfg = load_config("configs/bi.yaml")
    results = retrieve_topk(model, index, criteria_map, [("p1", "c1")], cfg)
    assert results, "retrieval results should not be empty"
    gold_sents = [l.sent_id for l in data["labels_sc"] if l.label == 1 and l.post_id == "p1" and l.cid == "c1"]
    recall = recall_at_k(gold_sents, results[0].candidates, k=5)
    assert recall == 1.0
