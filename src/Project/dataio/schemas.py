"""Pydantic schemas for dataset artifacts."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_binary(value: int) -> int:
    if value not in (0, 1):
        raise ValueError("label must be 0 or 1")
    return value


def _validate_prob(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must be in [0,1]")
    return value


class Post(BaseModel):
    """Reddit-like post."""

    post_id: str
    text: str
    subreddit: Optional[str] = None
    created_utc: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class Sentence(BaseModel):
    """Sentence belonging to a post."""

    post_id: str
    sent_id: str
    text: str
    char_span: Optional[Tuple[int, int]] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_span(self) -> "Sentence":
        if self.char_span:
            start, end = self.char_span
            if start < 0 or end < 0 or end < start:
                raise ValueError("char_span must be non-negative and end>=start")
        return self


class Criterion(BaseModel):
    """Diagnostic criterion."""

    cid: str
    name: str
    is_core: bool
    desc: str

    model_config = ConfigDict(extra="forbid")


class LabelSC(BaseModel):
    """Sentence–Criterion label."""

    post_id: str
    sent_id: str
    cid: str
    label: int
    rationale: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    _label_validator = field_validator("label", mode="before")(_validate_binary)


class LabelPC(BaseModel):
    """Post–Criterion label."""

    post_id: str
    cid: str
    label: int

    model_config = ConfigDict(extra="forbid")

    _label_validator = field_validator("label", mode="before")(_validate_binary)


class RetrievalSC(BaseModel):
    """Top-K retrieval result per post/criterion."""

    post_id: str
    cid: str
    candidates: List[Tuple[str, float, dict | None]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    def normalize_candidates(cls, values: dict) -> dict:
        raw = values.get("candidates", [])
        normalized = []
        for cand in raw:
            if isinstance(cand, (list, tuple)):
                if len(cand) >= 2:
                    sid, score = cand[0], cand[1]
                    meta = cand[2] if len(cand) >= 3 else {}
                    normalized.append((sid, float(score), meta or {}))
            elif isinstance(cand, dict) and {"sent_id", "score"} <= set(cand.keys()):
                normalized.append((cand["sent_id"], float(cand["score"]), cand.get("meta", {})))
        values["candidates"] = normalized
        return values

    @model_validator(mode="after")
    def validate_unique(self) -> "RetrievalSC":
        seen = set()
        for sent_id, *_ in self.candidates:
            if sent_id in seen:
                raise ValueError("duplicate sent_id in candidates")
            seen.add(sent_id)
        return self


class CESCSCore(BaseModel):
    """Cross-encoder S–C score."""

    post_id: str
    cid: str
    sent_id: str
    logit: float
    prob: float

    model_config = ConfigDict(extra="forbid")

    _prob_validator = field_validator("prob", mode="before")(_validate_prob)


class CEPCScore(BaseModel):
    """Cross-encoder P–C chunk score."""

    post_id: str
    cid: str
    chunk_id: str
    logit: float
    prob: float

    model_config = ConfigDict(extra="forbid")

    _prob_validator = field_validator("prob", mode="before")(_validate_prob)


class CEPCAggregated(BaseModel):
    """Post-level aggregated P–C score."""

    post_id: str
    cid: str
    agg: str
    logit: float
    prob: float

    model_config = ConfigDict(extra="forbid")

    _prob_validator = field_validator("prob", mode="before")(_validate_prob)


SchemaType = Sequence[
    Post
    | Sentence
    | Criterion
    | LabelSC
    | LabelPC
    | RetrievalSC
    | CESCSCore
    | CEPCScore
    | CEPCAggregated
]
