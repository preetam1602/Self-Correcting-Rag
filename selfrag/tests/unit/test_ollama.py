import asyncio
import json

import httpx
import pytest

from selfrag.providers import ollama
from selfrag.providers.ollama import ModelNotInstalled, OllamaClient


class FakeOllama:
    """Answers /api/chat per model from a script: an int is that HTTP status, a str the
    reply. Records every request; never talks to a real server."""

    def __init__(self, script: dict[str, list], installed=("primary:latest", "backup:latest")):
        self.script, self.installed, self.calls = script, installed, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert "/api/pull" not in request.url.path, "models must never be pulled"
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.installed]})
        body = json.loads(request.content)
        self.calls.append(body)
        outcome = self.script[body["model"]].pop(0)
        if isinstance(outcome, int):
            return httpx.Response(outcome, json={"error": "x"})
        return httpx.Response(200, json={"message": {"content": outcome},
                                         "prompt_eval_count": 7, "eval_count": 3})


def client(script, installed=("primary:latest", "backup:latest"), **kw):
    c = OllamaClient("primary", **kw)
    fake = FakeOllama(script, installed)
    c._client = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(fake))
    return c, fake


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None
    monkeypatch.setattr(ollama.asyncio, "sleep", instant)


def complete(c: OllamaClient, **kw):
    return asyncio.run(c.complete("q", max_tokens=5, **kw))


def test_reply_tokens_and_cpu_options():
    c, fake = client({"primary": ["hi"]}, num_thread=2, num_ctx=4096)
    out = complete(c, stop=["\n"])
    assert (out.text, out.input_tokens, out.output_tokens) == ("hi", 7, 3)
    body = fake.calls[0]
    assert body["stream"] is False
    assert body["options"] == {"num_thread": 2, "num_ctx": 4096, "temperature": 0.0,
                               "num_predict": 5, "stop": ["\n"]}


def test_thread_count_defaults_to_half_the_cores(monkeypatch):
    monkeypatch.setattr(ollama.os, "cpu_count", lambda: 8)
    c, fake = client({"primary": ["a"]})
    complete(c)
    assert fake.calls[0]["options"]["num_thread"] == 4


def test_missing_model_is_reported_not_pulled():
    c, fake = client({}, installed=["backup:latest"])
    with pytest.raises(ModelNotInstalled, match="ollama pull primary"):
        asyncio.run(c.check_installed())
    c, fake = client({"primary": [404]})
    with pytest.raises(ModelNotInstalled):
        complete(c)


def test_installed_models_match_with_or_without_latest_tag():
    c, _ = client({}, fallback_models=["backup"])
    asyncio.run(c.check_installed())


def test_busy_primary_fails_over_and_cools_down():
    c, fake = client({"primary": [503], "backup": ["a", "b"]}, fallback_models=["backup"])
    assert complete(c).text == "a"
    assert complete(c).text == "b"
    assert [b["model"] for b in fake.calls] == ["primary", "backup", "backup"]


def test_last_model_retries_then_raises():
    c, fake = client({"primary": [503, 503, 503]}, max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        complete(c)
    assert len(fake.calls) == 3


def test_think_is_dropped_when_rejected():
    c, fake = client({"primary": [400, "a", "b"]}, thinking_level="low")
    assert complete(c).text == "a"
    assert complete(c).text == "b"
    assert ["think" in b for b in fake.calls] == [True, False, False]


def test_logprobs_are_refused():
    c, _ = client({})
    with pytest.raises(ValueError):
        complete(c, logprobs=True)
