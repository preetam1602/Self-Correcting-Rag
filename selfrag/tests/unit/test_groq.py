import asyncio
from types import SimpleNamespace

import groq as groq_sdk
import httpx
import pytest

from selfrag.providers import groq
from selfrag.providers.groq import GroqClient


def api_error(code: int) -> groq_sdk.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq_sdk.APIStatusError("x", response=httpx.Response(code, request=request), body=None)


class FakeCompletions:
    """Answers per model from a script: an int raises that API error, a str is the reply."""

    def __init__(self, script: dict[str, list]):
        self.script, self.calls = script, []

    async def create(self, *, model, messages, max_completion_tokens, temperature, stop, **extra):
        self.calls.append((model, extra.get("reasoning_effort"), max_completion_tokens))
        outcome = self.script[model].pop(0)
        if isinstance(outcome, int):
            raise api_error(outcome)
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def client(script, **kw) -> tuple[GroqClient, FakeCompletions]:
    c = GroqClient("k", "primary", **kw)
    fake = FakeCompletions(script)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return c, fake


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None
    monkeypatch.setattr(groq.asyncio, "sleep", instant)


def complete(c: GroqClient) -> str:
    return asyncio.run(c.complete("q", max_tokens=5)).text


def test_overloaded_primary_fails_over_and_cools_down():
    c, fake = client({"primary": [503], "backup": ["a", "b"]}, fallback_models=["backup"])
    assert complete(c) == "a"
    assert complete(c) == "b"                       # primary is skipped while cooling down
    assert [m for m, _, _ in fake.calls] == ["primary", "backup", "backup"]


def test_primary_is_retried_after_cooldown():
    c, fake = client({"primary": [429, "p"], "backup": ["a"]}, fallback_models=["backup"],
                     fallback_cooldown_s=0)
    assert complete(c) == "a"
    assert complete(c) == "p"


def test_last_model_retries_then_raises():
    c, fake = client({"primary": [503, 503, 503]}, max_retries=2)
    with pytest.raises(groq_sdk.APIStatusError):
        complete(c)
    assert len(fake.calls) == 3


def test_non_retryable_error_raises_without_fallback():
    c, fake = client({"primary": [401], "backup": ["a"]}, fallback_models=["backup"])
    with pytest.raises(groq_sdk.APIStatusError):
        complete(c)
    assert [m for m, _, _ in fake.calls] == ["primary"]


def test_reasoning_adds_headroom_and_is_dropped_when_rejected():
    c, fake = client({"primary": [400, "a", "b"]}, thinking_level="minimal",
                     reasoning_headroom=100)
    assert complete(c) == "a"
    assert complete(c) == "b"
    assert [(e, t) for _, e, t in fake.calls] == [("low", 105), (None, 5), (None, 5)]


def test_logprobs_are_refused():
    c, _ = client({})
    with pytest.raises(ValueError):
        asyncio.run(c.complete("q", max_tokens=1, logprobs=True))
