from Project.dataio.loaders import PairExample, SimpleTokenizer, collate_pairwise_ce, sentence_chunker


def test_sentence_chunker_stride() -> None:
    text = " ".join([f"tok{i}" for i in range(30)])
    chunks = sentence_chunker(text, max_len=10, stride=5)
    assert len(chunks) >= 3
    assert chunks[1]["start"] == 5
    assert chunks[2]["start"] == 10


def test_truncation_only_first_keeps_criterion() -> None:
    tokenizer = SimpleTokenizer()
    example = PairExample(
        text=" ".join(["x"] * 40),
        criterion="criterion should not be truncated",
        label=1,
        meta={},
    )
    batch = collate_pairwise_ce([example], tokenizer=tokenizer, max_length=15)
    token_type_ids = batch["token_type_ids"][0]
    crit_tokens = len("criterion should not be truncated".split())
    crit_tokens_in_batch = int((token_type_ids == 1).sum().item())
    # criterion tokens plus final SEP
    assert crit_tokens_in_batch == crit_tokens + 1
