"""Groq LLMClient adapter with a concurrency cap, retries on rate limits / overload,
fallback models while the primary is unavailable, and a configurable reasoning effort."""
import asyncio
import logging
import random
import time
from typing import Sequence

import groq

from selfrag.providers.interfaces import Completion
from selfrag.providers.usage import record

log = logging.getLogger(__name__)

_RETRYABLE = {429, 500, 502, 503, 504}
# selfrag thinking levels -> Groq reasoning_effort (gpt-oss accepts low | medium | high)
_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}
_MAX_STOP = 4       # Groq accepts at most 4 stop sequences


def _status(exc: Exception) -> int | None:
    if isinstance(exc, groq.APIStatusError):
        return exc.status_code
    if isinstance(exc, (groq.APITimeoutError, groq.APIConnectionError)):
        return 503      # treat transport failures as transient overload
    return None


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Server-suggested Retry-After when given, else jittered exponential backoff."""
    if isinstance(exc, groq.APIStatusError):
        try:
            return float(exc.response.headers.get("retry-after", "")) + random.uniform(0, 1)
        except ValueError:
            pass
    return min(2 ** attempt, 30) + random.uniform(0, 1)


class GroqClient:
    supports_logprobs = False       # Groq does not return logprobs

    def __init__(self, api_key: str, model: str, *, fallback_models: Sequence[str] = (),
                 thinking_level: str | None = None, fallback_cooldown_s: float = 60.0,
                 max_concurrency: int = 4, max_retries: int = 5, timeout_s: float = 60.0,
                 reasoning_headroom: int = 1024):
        self.model = model
        self._models = [model, *dict.fromkeys(m for m in fallback_models if m != model)]
        self._effort = _EFFORT.get(thinking_level) if thinking_level else None
        # Reasoning tokens count against max_completion_tokens: without headroom they can use
        # up a small budget (critic labels) and leave the answer empty.
        self._headroom = reasoning_headroom
        self._no_reasoning: set[str] = set()    # models that rejected the reasoning params
        self._cooling: dict[str, float] = {}    # model -> monotonic time to try it again
        self._cooldown_s = fallback_cooldown_s
        self._client = groq.AsyncGroq(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries

    async def complete(self, prompt: str, *, max_tokens: int, temperature: float = 0.0,
                       stop: list[str] | None = None, logprobs: bool = False,
                       top_logprobs: int = 0) -> Completion:
        if logprobs:
            raise ValueError(f"{self.model} does not return logprobs")
        async with self._semaphore:
            response = await self._generate_with_fallback(prompt, max_tokens, temperature, stop)
        usage = response.usage
        completion = Completion(text=response.choices[0].message.content or "",
                                input_tokens=(usage and usage.prompt_tokens) or 0,
                                # reasoning tokens are included in completion_tokens
                                output_tokens=(usage and usage.completion_tokens) or 0)
        record(completion)
        return completion

    def _available_models(self) -> list[str]:
        now = time.monotonic()
        return [m for m in self._models if self._cooling.get(m, 0.0) <= now] or self._models

    async def _generate_with_fallback(self, prompt: str, max_tokens: int, temperature: float,
                                      stop: list[str] | None):
        models = self._available_models()
        for i, model in enumerate(models):
            last = i == len(models) - 1
            try:
                # Fail over at once while a fallback remains; only the last model retries.
                return await self._generate(model, prompt, max_tokens, temperature, stop,
                                            retries=self._max_retries if last else 0)
            except groq.APIError as exc:
                if last or _status(exc) not in _RETRYABLE:
                    raise
                self._cooling[model] = time.monotonic() + self._cooldown_s
                log.warning("%s unavailable (%s); falling back to %s for %.0fs",
                            model, _status(exc), models[i + 1], self._cooldown_s)

    async def _generate(self, model: str, prompt: str, max_tokens: int, temperature: float,
                        stop: list[str] | None, retries: int):
        attempt = 0
        while True:
            reasoning = self._effort is not None and model not in self._no_reasoning
            extra = {"reasoning_effort": self._effort, "include_reasoning": False} \
                if reasoning else {}
            try:
                return await self._client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens + (self._headroom if reasoning else 0),
                    temperature=temperature, stop=stop[:_MAX_STOP] if stop else None, **extra)
            except groq.APIError as exc:
                if _status(exc) == 400 and reasoning:
                    log.warning("%s rejected reasoning_effort; using its default", model)
                    self._no_reasoning.add(model)
                    continue
                if _status(exc) not in _RETRYABLE or attempt == retries:
                    raise
                await asyncio.sleep(_retry_delay(exc, attempt))
                attempt += 1
