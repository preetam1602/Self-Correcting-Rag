"""Per-request decision trace for admins: passages, critic probabilities and scores.

Collected only inside track_debug(), the same way usage is counted, so normal requests
pay nothing for it.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Iterator

from selfrag.models import Chunk

PREVIEW_CHARS = 300


@dataclass
class DebugTrace:
    config: dict = field(default_factory=dict)
    rewrites: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    final: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


_current: ContextVar[DebugTrace | None] = ContextVar("selfrag_debug", default=None)


def current() -> DebugTrace | None:
    return _current.get()


@contextmanager
def track_debug(enabled: bool = True) -> Iterator[DebugTrace | None]:
    """Record the decision trace of everything run inside this block (when enabled)."""
    trace = DebugTrace() if enabled else None
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


def passage_info(chunk: Chunk | None) -> dict | None:
    if chunk is None:
        return None
    return {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "title": chunk.title,
            "page": chunk.page, "section": chunk.section, "preview": chunk.text[:PREVIEW_CHARS]}


def probs(p: dict[str, float] | None) -> dict[str, float] | None:
    return {k: round(v, 3) for k, v in p.items()} if p else None
