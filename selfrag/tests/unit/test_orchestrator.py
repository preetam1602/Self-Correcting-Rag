"""The full Self-RAG graph against a deterministic fake LLM and retriever."""
import asyncio
import json
import re
from dataclasses import replace
from datetime import datetime, timezone

from selfrag.critics.critic import Critic
from selfrag.models import Beam, Chunk, Filter
from selfrag.orchestrator.graph import NOT_FOUND_ANSWER, SelfRAG, SelfRAGConfig
from selfrag.providers.interfaces import Completion
from selfrag.providers.usage import record

BASE = Chunk("", "d", "", "", None, None, "file:///x", "t", "c", [], "h",
             datetime(2026, 1, 1, tzinfo=timezone.utc))
GOOD = replace(BASE, chunk_id="good", title="Good doc", text="GOOD facts about notice periods.")
BAD = replace(BASE, chunk_id="bad", title="Bad doc", text="Unrelated cafeteria menu.")


class FakeLLM:
    """Answers each prompt type by its wording; GOOD evidence is relevant, anything else is not."""
    supports_logprobs = False

    def __init__(self, retrieve_decision="Continue", sentences=2):
        self.retrieve_decision, self.sentences = retrieve_decision, sentences
        self.prompts = []

    def _critic(self, p):
        evidence = " ".join(re.findall(r"<evidence>(.*?)</evidence>", p, re.DOTALL))
        if "Decide what the NEXT sentence" in p:
            return {"Yes": 0.0, "No": 0.0, "Continue": 0.0, self.retrieve_decision: 1.0}
        if "evidence is relevant" in p:
            return {"Relevant": 0.9, "Irrelevant": 0.1} if "GOOD" in evidence else \
                {"Relevant": 0.05, "Irrelevant": 0.95}
        if "supported by the evidence" in p:
            return {"Fully": 0.8, "Partially": 0.2, "No": 0.0}
        if "Rate how helpful" in p:
            return {"1": 0, "2": 0, "3": 0, "4": 1, "5": 0}
        raise AssertionError(p)

    async def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        out = await self._reply(prompt)
        record(Completion(out.text, input_tokens=10, output_tokens=2))
        return out

    async def _reply(self, p):
        if "Reply with JSON only" in p:
            return Completion(json.dumps(self._critic(p)))
        if "Reply with one word" in p:                      # critic in sample mode
            probs = self._critic(p)
            return Completion(max(probs, key=probs.get))
        if "different search queries" in p:
            return Completion("alt query")
        if "Rewrite the question" in p:
            return Completion("rewritten question")
        title = re.search(r"Evidence \[\d+\] \((.*?)\)", p)
        if "Answer the question in one to three sentences, using only" in p:
            return Completion(f"Answer from {title.group(1)}. [1]" if "GOOD" in p else "<END>")
        if "Write the NEXT single sentence" in p:
            done = len(re.findall(r"Sentence \d", p.split("Answer so far:")[1]))
            if done >= self.sentences:
                return Completion("<END>")
            source = title.group(1) if title else "no evidence"
            return Completion(f"Sentence {done + 1} from {source}.\nextra line")
        raise AssertionError(p)


class FakeRetriever:
    def __init__(self, passages):
        self.passages, self.calls = passages, []

    async def retrieve(self, query, filters, top_n, variants=()):
        self.calls.append((query, list(variants), filters))
        return [(p, 0.5) for p in self.passages][:top_n]


def run(passages, cfg, **llm_kw):
    llm, retriever = FakeLLM(**llm_kw), FakeRetriever(passages)
    rag = SelfRAG(Critic(llm, "verbal"), llm, llm, retriever, cfg)
    state = asyncio.run(rag.answer("What is the notice period?", Filter(tenant_id="t", collection="hr")))
    return state, llm, retriever


def generation_prompts(llm):
    return [p for p in llm.prompts if "Answer the question" in p or "NEXT single sentence" in p]


def test_answer_mode_cites_the_relevant_passage_only():
    state, llm, retriever = run([BAD, GOOD], SelfRAGConfig(segment_unit="answer"))
    assert state["status"] == "ok"
    assert state["answer"] == "Answer from Good doc."          # citation mark stripped
    assert state["best"].citations == [GOOD]
    assert len(generation_prompts(llm)) == 1                    # BAD skipped by IsRel filter
    query, variants, filters = retriever.calls[0]
    assert variants == ["alt query"] and filters == Filter(tenant_id="t", collection="hr")


def test_nothing_relevant_rewrites_once_then_falls_back():
    state, llm, retriever = run([BAD], SelfRAGConfig(segment_unit="answer", max_rewrites=1))
    assert (state["status"], state["answer"], state["best"]) == ("not_found", NOT_FOUND_ANSWER, None)
    assert [c[0] for c in retriever.calls] == ["What is the notice period?", "rewritten question"]


def test_sentence_mode_continue_reuses_passage_and_ends():
    state, llm, retriever = run([GOOD], SelfRAGConfig(segment_unit="sentence"),
                                retrieve_decision="Continue", sentences=2)
    assert state["status"] == "ok"
    assert state["best"].segments == ["Sentence 1 from Good doc.", "Sentence 2 from Good doc."]
    assert state["best"].citations == [GOOD, GOOD]
    assert len(retriever.calls) == 1                            # only the forced first retrieval


def test_sentence_mode_retrieve_yes_searches_with_previous_segment():
    state, llm, retriever = run([GOOD], SelfRAGConfig(segment_unit="sentence"),
                                retrieve_decision="Yes", sentences=2)
    assert len(retriever.calls) == 3                            # t = 0, 1, 2 (the last one ends)
    assert retriever.calls[1][0] == "What is the notice period? Sentence 1 from Good doc."


def test_sentence_mode_no_retrieve_writes_without_evidence():
    state, *_ = run([GOOD], SelfRAGConfig(segment_unit="sentence"),
                    retrieve_decision="No", sentences=2)
    assert state["best"].segments == ["Sentence 1 from Good doc.", "Sentence 2 from no evidence."]
    assert state["best"].citations == [GOOD, None]


def test_segment_cap_finalizes():
    state, *_ = run([GOOD], SelfRAGConfig(segment_unit="sentence", max_segments=3),
                    retrieve_decision="Continue", sentences=99)
    assert state["status"] == "ok" and len(state["best"].segments) == 3


def test_route_branches():
    rag = SelfRAG(Critic(FakeLLM()), FakeLLM(), FakeLLM(), FakeRetriever([]),
                  SelfRAGConfig(max_segments=8, max_rewrites=1))
    live, ended = Beam(["a"], [None], [1.0]), Beam(["a"], [None], [1.0], done=True)
    assert rag.route({"beams": [], "t": 1, "rewrites": 0}) == "rewrite"
    assert rag.route({"beams": [Beam(done=True)], "t": 1, "rewrites": 1}) == "fallback"
    assert rag.route({"beams": [live], "t": 1}) == "step"
    assert rag.route({"beams": [live, ended], "t": 8}) == "finalize"
    assert rag.route({"beams": [ended], "t": 2}) == "finalize"


def respond(passages, cfg, **llm_kw):
    llm = FakeLLM(**llm_kw)
    rag = SelfRAG(Critic(llm, "verbal"), llm, llm, FakeRetriever(passages), cfg)
    return asyncio.run(rag.respond("What is the notice period?", Filter(tenant_id="t"))), llm


def test_respond_ok_has_citations_trace_id_and_usage():
    resp, llm = respond([GOOD], SelfRAGConfig(segment_unit="sentence"),
                        retrieve_decision="Continue", sentences=2)
    assert resp.status == "ok"
    assert resp.answer == "Sentence 1 from Good doc [1]. Sentence 2 from Good doc [1]."
    assert [(s.text, s.citation.n, s.citation.title, s.citation.support) for s in resp.segments] == [
        ("Sentence 1 from Good doc.", 1, "Good doc", "fully"),
        ("Sentence 2 from Good doc.", 1, "Good doc", "fully")]
    assert resp.trace_id.startswith("tr_") and len(resp.trace_id) == 35
    assert resp.usage.llm_calls == len(llm.prompts) > 0      # every call counted, across tasks
    assert resp.usage.input_tokens == 10 * resp.usage.llm_calls
    assert resp.usage.latency_ms >= 0


def test_respond_not_found_has_no_segments():
    resp, _ = respond([BAD], SelfRAGConfig(segment_unit="answer"))
    assert (resp.status, resp.answer, resp.segments) == ("not_found", NOT_FOUND_ANSWER, [])
    assert resp.usage.llm_calls > 0


def test_each_request_gets_its_own_trace_id():
    a, _ = respond([GOOD], SelfRAGConfig())
    b, _ = respond([GOOD], SelfRAGConfig())
    assert a.trace_id != b.trace_id


# ---------------------------------------------------------------- debug trace (admin)

import pytest

from selfrag.config.overrides import OverridesNotAllowed


def debug_respond(passages, cfg, **llm_kw):
    llm = FakeLLM(**llm_kw)
    rag = SelfRAG(Critic(llm, "verbal"), llm, llm, FakeRetriever(passages), cfg)
    return asyncio.run(rag.respond("What is the notice period?", Filter(tenant_id="t"),
                                   is_admin=True, debug_trace=True))


def test_debug_trace_shows_passages_candidates_and_final_choice():
    resp = debug_respond([BAD, GOOD], SelfRAGConfig(segment_unit="answer", w_sup=1.5))
    trace = resp.debug
    assert trace["config"]["w_sup"] == 1.5 and trace["rewrites"] == []
    step = trace["steps"][0]
    assert step["decision"] == "retrieve" and step["variants"] == ["alt query"]
    assert [(p["chunk_id"], p["rerank_score"], p["kept"]) for p in step["passages"]] == [
        ("bad", 0.5, False), ("good", 0.5, True)]
    assert step["passages"][1]["rel_probs"] == {"Relevant": 0.9, "Irrelevant": 0.1}
    [cand] = step["candidates"]                                 # BAD was never expanded
    assert cand["chunk_id"] == "good" and cand["s_sup"] == 0.9
    assert cand["segment_score"] == round(0.9 + 1.5 * 0.9, 3)
    [final] = trace["final"]
    assert final["chosen"] and final["s_use"] == 0.5
    assert final["final_score"] == round(final["mean_segment_score"] + 0.5 * 0.5, 3)


def test_debug_trace_records_retrieve_critic_and_rewrites():
    trace = debug_respond([GOOD], SelfRAGConfig(segment_unit="sentence"),
                          retrieve_decision="Continue", sentences=1).debug
    assert trace["steps"][1]["retrieve_probs"]["Continue"] == 1.0
    assert trace["steps"][1]["decision"] == "continue"
    trace = debug_respond([BAD], SelfRAGConfig(segment_unit="answer", max_rewrites=1)).debug
    assert trace["rewrites"] == ["rewritten question"] and trace["final"] == []


def test_no_debug_trace_unless_asked_and_admin_only():
    resp, _ = respond([GOOD], SelfRAGConfig())
    assert resp.debug is None
    rag = SelfRAG(Critic(FakeLLM()), FakeLLM(), FakeLLM(), FakeRetriever([GOOD]), SelfRAGConfig())
    with pytest.raises(OverridesNotAllowed):
        asyncio.run(rag.respond("q", Filter(tenant_id="t"), debug_trace=True))
