"""Ollama LLMClient adapter for models served locally, with CPU-friendly limits.

It never downloads models: a model that is not installed is reported with the
`ollama pull` command to run by hand. Concurrency defaults to one request at a time and
the thread count is capped, so inference leaves the rest of the machine responsive."""
import asyncio
import logging
import os
import random
import time
from typing import Sequence

import httpx

from selfrag.providers.interfaces import Completion
from selfrag.providers.usage import record

log = logging.getLogger(__name__)

_RETRYABLE = {429, 500, 502, 503, 504}
# selfrag thinking levels -> Ollama `think` (bool, or low | medium | high for gpt-oss)
_THINK = {"minimal": False, "low": "low", "medium": "medium", "high": "high"}


class ModelNotInstalled(RuntimeError):
    def __init__(self, model: str):
        super().__init__(f"Ollama model {model!r} is not installed. Download it yourself "
                         f"with `ollama pull {model}`; selfrag never pulls models.")
        self.model = model


def default_num_thread() -> int:
    """Half the logical cores, so inference does not saturate the CPU."""
    return max(1, (os.cpu_count() or 2) // 2)


def _status(exc: Exception) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if isinstance(exc, httpx.TransportError):
        return 503      # server not up yet, or busy loading a model
    return None


class OllamaClient:
    supports_logprobs = False

    def __init__(self, model: str, *, base_url: str = "http://localhost:11434",
                 fallback_models: Sequence[str] = (), thinking_level: str | None = None,
                 fallback_cooldown_s: float = 60.0, max_concurrency: int = 1,
                 max_retries: int = 2, timeout_s: float = 300.0,
                 num_thread: int | None = None, num_ctx: int | None = None,
                 keep_alive: str = "5m"):
        self.model = model
        self._models = [model, *dict.fromkeys(m for m in fallback_models if m != model)]
        self._think = _THINK.get(thinking_level) if thinking_level else None
        self._no_think: set[str] = set()        # models that rejected the think param
        self._cooling: dict[str, float] = {}    # model -> monotonic time to try it again
        self._cooldown_s = fallback_cooldown_s
        self._options = {"num_thread": num_thread or default_num_thread()}
        if num_ctx:
            self._options["num_ctx"] = num_ctx
        self._keep_alive = keep_alive
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries

    async def check_installed(self) -> None:
        """Raise ModelNotInstalled unless every configured model is available locally.
        Only lists installed models: nothing is loaded into memory or downloaded."""
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        installed = {m["name"] for m in response.json().get("models", [])}
        installed |= {n.removesuffix(":latest") for n in installed}
        for model in self._models:
            if model not in installed:
                raise ModelNotInstalled(model)

    async def complete(self, prompt: str, *, max_tokens: int, temperature: float = 0.0,
                       stop: list[str] | None = None, logprobs: bool = False,
                       top_logprobs: int = 0) -> Completion:
        if logprobs:
            raise ValueError(f"{self.model} does not return logprobs")
        async with self._semaphore:
            body = await self._generate_with_fallback(prompt, max_tokens, temperature, stop)
        completion = Completion(text=body.get("message", {}).get("content") or "",
                                input_tokens=body.get("prompt_eval_count") or 0,
                                output_tokens=body.get("eval_count") or 0)
        record(completion)
        return completion

    async def close(self) -> None:
        await self._client.aclose()

    def _available_models(self) -> list[str]:
        now = time.monotonic()
        return [m for m in self._models if self._cooling.get(m, 0.0) <= now] or self._models

    async def _generate_with_fallback(self, prompt: str, max_tokens: int, temperature: float,
                                      stop: list[str] | None) -> dict:
        models = self._available_models()
        for i, model in enumerate(models):
            last = i == len(models) - 1
            try:
                # Fail over at once while a fallback remains; only the last model retries.
                return await self._generate(model, prompt, max_tokens, temperature, stop,
                                            retries=self._max_retries if last else 0)
            except httpx.HTTPError as exc:
                if last or _status(exc) not in _RETRYABLE:
                    raise
                self._cooling[model] = time.monotonic() + self._cooldown_s
                log.warning("%s unavailable (%s); falling back to %s for %.0fs",
                            model, _status(exc), models[i + 1], self._cooldown_s)

    async def _generate(self, model: str, prompt: str, max_tokens: int, temperature: float,
                        stop: list[str] | None, retries: int) -> dict:
        attempt = 0
        while True:
            think = self._think is not None and model not in self._no_think
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "keep_alive": self._keep_alive,
                       "options": {**self._options, "temperature": temperature,
                                   "num_predict": max_tokens,
                                   **({"stop": stop} if stop else {})}}
            if think:
                payload["think"] = self._think
            try:
                response = await self._client.post("/api/chat", json=payload)
                if response.status_code == 404:
                    raise ModelNotInstalled(model)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                if _status(exc) == 400 and think:
                    log.warning("%s rejected think; using its default", model)
                    self._no_think.add(model)
                    continue
                if _status(exc) not in _RETRYABLE or attempt == retries:
                    raise
                await asyncio.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                attempt += 1
