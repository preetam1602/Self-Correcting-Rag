"""Turns any chat LLM into a reflection-token predictor that returns label probabilities."""
import asyncio
import json
import math
import re
from collections import Counter

from selfrag.providers.interfaces import LLMClient

MODES = ("logprobs", "sample", "verbal")


def _options(labels: list[str]) -> str:
    return ", ".join(labels[:-1]) + " or " + labels[-1]


def match_label(text: str, labels: list[str]) -> str | None:
    t = text.strip().strip("*\"'`").lower()
    return next((l for l in labels if t.startswith(l.lower())), None)


def parse_verbal(text: str, labels: list[str]) -> dict[str, float] | None:
    """Read {"label": probability, ...} from the reply; None when it cannot be parsed."""
    found = re.search(r"\{.*\}", text, re.DOTALL)
    if not found:
        return None
    try:
        data = json.loads(found.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    lowered = {str(k).strip().lower(): v for k, v in data.items()}
    raw = {}
    for label in labels:
        try:
            raw[label] = max(float(lowered.get(label.lower(), 0.0)), 0.0)
        except (TypeError, ValueError):
            raw[label] = 0.0
    return raw if sum(raw.values()) > 0 else None


def _normalise(raw: dict[str, float]) -> dict[str, float]:
    z = sum(raw.values())
    return {l: v / z for l, v in raw.items()}


class Critic:
    def __init__(self, llm: LLMClient, mode: str = "verbal", n_samples: int = 5):
        if mode not in MODES:
            raise ValueError(f"critic mode must be one of {MODES}")
        if mode == "logprobs" and not llm.supports_logprobs:
            raise ValueError("critic mode 'logprobs' needs an LLM that returns logprobs")
        self.llm, self.mode, self.n = llm, mode, n_samples

    async def label_probs(self, prompt: str, labels: list[str]) -> dict[str, float]:
        one_word = f"{prompt}\nReply with one word: {_options(labels)}."
        if self.mode == "logprobs":
            out = await self.llm.complete(one_word, max_tokens=1, logprobs=True, top_logprobs=20)
            top = {t.strip().lower(): lp for t, lp in (out.top_logprobs or [{}])[0].items()}
            raw = {l: math.exp(top.get(l.lower(), -30.0)) for l in labels}
        elif self.mode == "sample":
            outs = await asyncio.gather(*[self.llm.complete(one_word, max_tokens=5, temperature=0.7)
                                          for _ in range(self.n)])
            votes = Counter(match_label(o.text, labels) or labels[-1] for o in outs)
            raw = {l: votes.get(l, 0) + 0.1 for l in labels}         # +0.1 smoothing
        else:  # verbal
            example = json.dumps({l: round(1 / len(labels), 2) for l in labels})
            out = await self.llm.complete(
                f"{prompt}\nReply with JSON only, giving the probability of each label "
                f"({_options(labels)}), summing to 1. Format: {example}",
                max_tokens=100)
            raw = parse_verbal(out.text, labels)
            if raw is None:                 # malformed reply: fall back to a bare label, else uniform
                label = match_label(out.text, labels)
                raw = {l: (1.0 if l == label else 0.1) for l in labels} if label else \
                    {l: 1.0 for l in labels}
        return _normalise(raw)
