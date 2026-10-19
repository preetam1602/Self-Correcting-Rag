"""Per-request LLM usage accounting. LLMClient adapters call record() after each completion."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from selfrag.providers.interfaces import Completion


@dataclass
class UsageCounter:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


_current: ContextVar[UsageCounter | None] = ContextVar("llm_usage", default=None)


@contextmanager
def track_usage() -> Iterator[UsageCounter]:
    """Count every LLM call made inside this block, including in tasks it spawns."""
    counter = UsageCounter()
    token = _current.set(counter)
    try:
        yield counter
    finally:
        _current.reset(token)


def record(completion: Completion) -> None:
    counter = _current.get()
    if counter is not None:
        counter.llm_calls += 1
        counter.input_tokens += completion.input_tokens
        counter.output_tokens += completion.output_tokens
