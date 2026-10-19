"""Gemini LLMClient adapter with a concurrency cap, retries on rate limits / overload,
fallback models while the primary is unavailable, and a configurable thinking level."""
import asyncio
import logging
import random
import re
import time
from typing import Sequence

from google import genai
from google.genai import errors, types

from selfrag.providers.interfaces import Completion
from selfrag.providers.usage import record

log = logging.getLogger(__name__)

_RETRYABLE = {429, 500, 503, 504}


def _retry_delay(exc: errors.APIError, attempt: int) -> float:
    """Server-suggested delay when given (e.g. '12s'), else jittered exponential backoff."""
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", str(exc.details))
    if match:
        return float(match.group(1)) + random.uniform(0, 1)
    return min(2 ** attempt, 30) + random.uniform(0, 1)


class GeminiClient:
    supports_logprobs = False       # no Gemini model available to this key returns logprobs

    def __init__(self, api_key: str, model: str, *, fallback_models: Sequence[str] = (),
                 thinking_level: str | None = None, fallback_cooldown_s: float = 60.0,
                 max_concurrency: int = 4, max_retries: int = 5, timeout_s: float = 60.0):
        self.model = model
        self._models = [model, *dict.fromkeys(m for m in fallback_models if m != model)]
        # Thinking tokens count against max_output_tokens: left at the model default they
        # can use up a small budget (critic labels) and leave the answer empty.
        self._thinking = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level \
            else None
        self._no_thinking: set[str] = set()     # models that rejected the thinking config
        self._cooling: dict[str, float] = {}    # model -> monotonic time to try it again
        self._cooldown_s = fallback_cooldown_s
        self._client = genai.Client(api_key=api_key,
                                    http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries

    async def complete(self, prompt: str, *, max_tokens: int, temperature: float = 0.0,
                       stop: list[str] | None = None, logprobs: bool = False,
                       top_logprobs: int = 0) -> Completion:
        if logprobs:
            raise ValueError(f"{self.model} does not return logprobs")
        async with self._semaphore:
            response = await self._generate_with_fallback(prompt, max_tokens, temperature, stop)
        usage = response.usage_metadata
        completion = Completion(text=response.text or "",
                                input_tokens=(usage and usage.prompt_token_count) or 0,
                                # thinking tokens are billed as output
                                output_tokens=((usage and usage.candidates_token_count) or 0)
                                + ((usage and usage.thoughts_token_count) or 0))
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
            except errors.APIError as exc:
                if last or exc.code not in _RETRYABLE:
                    raise
                self._cooling[model] = time.monotonic() + self._cooldown_s
                log.warning("%s unavailable (%s); falling back to %s for %.0fs",
                            model, exc.code, models[i + 1], self._cooldown_s)

    async def _generate(self, model: str, prompt: str, max_tokens: int, temperature: float,
                        stop: list[str] | None, retries: int):
        attempt = 0
        while True:
            thinking = None if model in self._no_thinking else self._thinking
            config = types.GenerateContentConfig(max_output_tokens=max_tokens,
                                                 temperature=temperature, stop_sequences=stop,
                                                 thinking_config=thinking)
            try:
                return await self._client.aio.models.generate_content(
                    model=model, contents=prompt, config=config)
            except errors.APIError as exc:
                if exc.code == 400 and thinking is not None:
                    log.warning("%s rejected thinking_level; using its default", model)
                    self._no_thinking.add(model)
                    continue
                if exc.code not in _RETRYABLE or attempt == retries:
                    raise
                await asyncio.sleep(_retry_delay(exc, attempt))
                attempt += 1
