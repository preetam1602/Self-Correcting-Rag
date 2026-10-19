"""Data models shared across pipeline stages."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class Document:
    doc_id: str
    tenant_id: str
    collection: str
    title: str
    source_url: str
    acl: list[str]


@dataclass(frozen=True)
class Filter:
    """Retrieval scope, applied inside the vector and sparse queries (never after retrieval).

    A chunk is visible when it belongs to `tenant_id` (and `collection`, if given) and either
    its ACL is empty (the whole tenant may read it) or it shares a group with `acl_groups`.
    """
    tenant_id: str
    collection: str | None = None
    acl_groups: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.tenant_id:
            raise ValueError("tenant_id is required for retrieval")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    section: str | None
    page: int | None
    source_url: str
    tenant_id: str
    collection: str
    acl: list[str]
    content_hash: str
    updated_at: datetime


@dataclass
class Candidate:
    """One candidate next segment y_t, with its critique label probabilities."""
    text: str
    passage: Chunk | None               # d, or None when generated without retrieval
    gen_logprob: float | None = None    # mean token log-prob of y_t, when the LLM exposes it
    rel: dict[str, float] | None = None
    sup: dict[str, float] | None = None
    use: dict[str, float] | None = None


@dataclass
class Beam:
    """A partial answer: its segments, the passage each one cites, and their scores."""
    segments: list[str] = field(default_factory=list)
    citations: list[Chunk | None] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    supports: list[str | None] = field(default_factory=list)   # IsSup label per segment
    last_passage: Chunk | None = None
    done: bool = False

    @property
    def score(self) -> float:
        return sum(self.scores) / max(len(self.scores), 1)


# ---------------------------------------------------------------- API response (F4)

class Citation(BaseModel):
    n: int
    doc_id: str
    title: str
    page: int | None
    url: str
    support: Literal["fully", "partially", "no"] | None


class AnswerSegment(BaseModel):
    text: str
    citation: Citation | None


class Usage(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class AnswerResponse(BaseModel):
    status: Literal["ok", "not_found"]
    answer: str
    segments: list[AnswerSegment]
    trace_id: str
    usage: Usage
    debug: dict | None = None           # decision trace; admins only, when requested
