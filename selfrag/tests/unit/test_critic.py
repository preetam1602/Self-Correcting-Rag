import asyncio
import math

import pytest

from selfrag.critics import critics
from selfrag.critics.critic import Critic, parse_verbal
from selfrag.providers.interfaces import Completion

LABELS = ["Fully", "Partially", "No"]


class ScriptedLLM:
    def __init__(self, replies, supports_logprobs=False, top=None):
        self.replies, self.supports_logprobs, self.top, self.prompts = list(replies), supports_logprobs, top, []

    async def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        return Completion(self.replies.pop(0) if self.replies else "", top_logprobs=self.top)


def probs(llm, mode="verbal", labels=LABELS):
    return asyncio.run(Critic(llm, mode, n_samples=5).label_probs("Judge.", labels))


def test_verbal_json():
    p = probs(ScriptedLLM(['{"Fully": 0.6, "Partially": 0.3, "No": 0.1}']))
    assert p == pytest.approx({"Fully": 0.6, "Partially": 0.3, "No": 0.1})


def test_verbal_json_in_code_fence_with_other_case_and_missing_label():
    p = probs(ScriptedLLM(['```json\n{"fully": 3, "no": 1}\n```']))
    assert p == pytest.approx({"Fully": 0.75, "Partially": 0.0, "No": 0.25})


def test_verbal_falls_back_to_bare_label_then_uniform():
    p = probs(ScriptedLLM(["Partially."]))
    assert max(p, key=p.get) == "Partially"
    assert probs(ScriptedLLM(["unclear"])) == pytest.approx({l: 1 / 3 for l in LABELS})


def test_parse_verbal_rejects_garbage():
    assert parse_verbal("{not json}", LABELS) is None
    assert parse_verbal('{"Fully": 0, "No": 0}', LABELS) is None
    assert parse_verbal('{"Fully": "high"}', LABELS) is None


def test_sample_mode_counts_votes_with_smoothing():
    p = probs(ScriptedLLM(["Fully", "Fully", "fully.", "Partially", "No"]), mode="sample")
    z = 3.1 + 1.1 + 1.1
    assert p == pytest.approx({"Fully": 3.1 / z, "Partially": 1.1 / z, "No": 1.1 / z})


def test_logprobs_mode_reads_first_token():
    top = [{"Fully": math.log(0.7), " Partially": math.log(0.2), "No": math.log(0.1)}]
    p = probs(ScriptedLLM([""], supports_logprobs=True, top=top), mode="logprobs")
    assert p == pytest.approx({"Fully": 0.7, "Partially": 0.2, "No": 0.1}, abs=1e-6)


def test_logprobs_mode_needs_support():
    with pytest.raises(ValueError):
        Critic(ScriptedLLM([]), "logprobs")


def test_critic_prompts_fill_every_field():
    llm = ScriptedLLM(['{"Relevant": 1, "Irrelevant": 0}'])
    asyncio.run(critics.isrel(Critic(llm), "Q?", "", "Passage text"))
    assert "Q?" in llm.prompts[0] and "Passage text" in llm.prompts[0]
    assert "(none)" in llm.prompts[0] and "{" in llm.prompts[0].split("Format:")[1]
