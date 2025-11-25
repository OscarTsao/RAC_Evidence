import json
from pathlib import Path

import pytest

from Project.dataio.schemas import (
    CESCSCore,
    CEPCScore,
    CEPCAggregated,
    Criterion,
    LabelPC,
    LabelSC,
    Post,
    RetrievalSC,
    Sentence,
)
from Project.utils.io import load_models


def test_raw_files_validate() -> None:
    raw_dir = Path("data/raw")
    posts = load_models(raw_dir / "posts.jsonl", Post)
    sentences = load_models(raw_dir / "sentences.jsonl", Sentence)
    labels_sc = load_models(raw_dir / "labels_sc.jsonl", LabelSC)
    labels_pc = load_models(raw_dir / "labels_pc.jsonl", LabelPC)
    criteria = [Criterion.model_validate(c) for c in json.loads((raw_dir / "criteria.json").read_text())]
    assert posts and sentences and labels_sc and labels_pc and criteria


def test_prob_validators() -> None:
    with pytest.raises(ValueError):
        CESCSCore(post_id="p", cid="c", sent_id="s", logit=0.0, prob=1.5)
    with pytest.raises(ValueError):
        CEPCScore(post_id="p", cid="c", chunk_id="ch", logit=0.0, prob=-0.1)
    with pytest.raises(ValueError):
        CEPCAggregated(post_id="p", cid="c", agg="max", logit=0.0, prob=2.0)


def test_retrieval_candidate_uniqueness() -> None:
    RetrievalSC(post_id="p1", cid="c1", candidates=[("s1", 0.5), ("s2", 0.4)])
    with pytest.raises(ValueError):
        RetrievalSC(post_id="p1", cid="c1", candidates=[("s1", 0.5), ("s1", 0.4)])
