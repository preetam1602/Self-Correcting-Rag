import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors

from selfrag.providers import gemini
from selfrag.providers.gemini import GeminiClient


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}})


class FakeModels:
    """Answers per model from a script: an int raises that API error, a str is the reply."""

    def __init__(self, script: dict[str, list]):
        self.script, self.calls = script, []

    async def generate_content(self, *, model, contents, config):
        self.calls.append((model, config.thinking_config))
        outcome = self.script[model].pop(0)
        if isinstance(outcome, int):
            raise api_error(outcome)
        return SimpleNamespace(text=outcome, usage_metadata=None)


def client(script, **kw) -> tuple[GeminiClient, FakeModels]:
    c = GeminiClient("k", "primary", **kw)
    fake = FakeModels(script)
    c._client = SimpleNamespace(aio=SimpleNamespace(models=fake))
    return c, fake


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None
    monkeypatch.setattr(gemini.asyncio, "sleep", instant)


def complete(c: GeminiClient) -> str:
    return asyncio.run(c.complete("q", max_tokens=5)).text


def test_overloaded_primary_fails_over_and_cools_down():
    c, fake = client({"primary": [503], "backup": ["a", "b"]}, fallback_models=["backup"])
    assert complete(c) == "a"
    assert complete(c) == "b"                       # primary is skipped while cooling down
    assert [m for m, _ in fake.calls] == ["primary", "backup", "backup"]


def test_primary_is_retried_after_cooldown():
    c, fake = client({"primary": [429, "p"], "backup": ["a"]}, fallback_models=["backup"],
                     fallback_cooldown_s=0)
    assert complete(c) == "a"
    assert complete(c) == "p"


def test_last_model_retries_then_raises():
    c, fake = client({"primary": [503, 503, 503]}, max_retries=2)
    with pytest.raises(errors.APIError):
        complete(c)
    assert len(fake.calls) == 3


def test_non_retryable_error_does_not_fail_over():
    c, fake = client({"primary": [401], "backup": ["a"]}, fallback_models=["backup"])
    with pytest.raises(errors.APIError):
        complete(c)
    assert [m for m, _ in fake.calls] == ["primary"]


def test_thinking_level_dropped_for_models_that_reject_it():
    c, fake = client({"primary": [400, "a", "b"]}, thinking_level="minimal")
    assert complete(c) == "a"
    assert complete(c) == "b"
    assert fake.calls[0][1].thinking_level.value == "MINIMAL"
    assert fake.calls[1][1] is None and fake.calls[2][1] is None
