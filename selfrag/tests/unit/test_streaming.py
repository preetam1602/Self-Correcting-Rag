import asyncio
import json
import re
from dataclasses import replace

from selfrag.api.sse import encode
from selfrag.critics.critic import Critic
from selfrag.models import AnswerSegment, Filter
from selfrag.orchestrator.graph import NOT_FOUND_ANSWER, SelfRAG, SelfRAGConfig
from selfrag.orchestrator.streaming import BeamSwitchEvent, StreamEvent, sync_leader
from selfrag.tests.unit.test_orchestrator import BAD, GOOD, FakeLLM, FakeRetriever

A, B, C = (AnswerSegment(text=t, citation=None) for t in "ABC")


def kinds(events):
    return [(e.event, getattr(e.data, "index", getattr(e.data, "from_index", None)))
            for e in events]


def test_sync_leader_sends_only_new_segments():
    sent = []
    assert kinds(sync_leader(sent, [A])) == [("segment", 0)]
    assert kinds(sync_leader(sent, [A, B])) == [("segment", 1)]
    assert sync_leader(sent, [A, B]) == []
    assert sent == [A, B]


def test_sync_leader_switches_from_first_difference():
    sent = [A, B]
    assert kinds(sync_leader(sent, [A, C, B])) == [("beam_switch", 1), ("segment", 1), ("segment", 2)]
    assert sent == [A, C, B]


def test_sync_leader_switch_to_shorter_leader():
    sent = [A, B]
    assert kinds(sync_leader(sent, [A])) == [("beam_switch", 1)]
    assert sent == [A]


def test_sse_encoding():
    text = encode(StreamEvent("beam_switch", BeamSwitchEvent(from_index=2)))
    assert text == 'event: beam_switch\ndata: {"from_index":2}\n\n'


def collect(rag, question="What is the notice period?"):
    async def run():
        return [e async for e in rag.stream(question, Filter(tenant_id="t"))]
    return asyncio.run(run())


def test_stream_sends_segments_then_done_matching_respond():
    llm = FakeLLM(retrieve_decision="Continue", sentences=2)
    rag = SelfRAG(Critic(llm), llm, llm, FakeRetriever([GOOD]), SelfRAGConfig(segment_unit="sentence"))
    events = collect(rag)
    assert kinds(events) == [("segment", 0), ("segment", 1), ("done", None)]
    done = events[-1].data
    assert done.status == "ok" and done.usage.llm_calls == len(llm.prompts)
    streamed = [(e.data.text, e.data.citation) for e in events[:-1]]
    assert streamed == [(s.text, s.citation) for s in done.segments]


class PrefersOtherAtTheEnd(FakeLLM):
    """GOOD wins each step (better IsSup), but IsUse makes OTHER's answer win at finalize."""

    def _critic(self, p):
        if "supported by the evidence" in p and "OTHER" in p:
            return {"Fully": 0.0, "Partially": 1.0, "No": 0.0}
        if "Rate how helpful" in p:
            return {"5": 1.0, "1": 0, "2": 0, "3": 0, "4": 0} if "Other doc" in p else \
                {"1": 1.0, "2": 0, "3": 0, "4": 0, "5": 0}
        return super()._critic(p)


def test_stream_switches_when_finalize_picks_another_beam():
    other = replace(GOOD, chunk_id="other", title="Other doc", text="GOOD and OTHER facts.")
    llm = PrefersOtherAtTheEnd()
    rag = SelfRAG(Critic(llm), llm, llm, FakeRetriever([GOOD, other]),
                  SelfRAGConfig(segment_unit="answer"))
    events = collect(rag)
    assert kinds(events) == [("segment", 0), ("beam_switch", 0), ("segment", 0), ("done", None)]
    assert events[0].data.text == "Answer from Good doc."
    assert events[2].data.text == "Answer from Other doc."
    assert events[-1].data.answer == "Answer from Other doc [1]."


def test_stream_not_found_sends_only_done():
    llm = FakeLLM()
    rag = SelfRAG(Critic(llm), llm, llm, FakeRetriever([BAD]), SelfRAGConfig(segment_unit="answer"))
    events = collect(rag)
    assert kinds(events) == [("done", None)]
    assert (events[0].data.status, events[0].data.answer) == ("not_found", NOT_FOUND_ANSWER)


class BrokenRetriever:
    async def retrieve(self, *a, **kw):
        raise ConnectionError("qdrant down at 10.0.0.5")


def test_stream_error_event_hides_internal_details():
    llm = FakeLLM()
    rag = SelfRAG(Critic(llm), llm, llm, BrokenRetriever(), SelfRAGConfig())
    events = collect(rag)
    assert kinds(events) == [("error", None)]
    payload = json.loads(encode(events[0]).split("data: ")[1])
    assert re.fullmatch(r"tr_[0-9a-f]{32}", payload["trace_id"])
    assert "10.0.0.5" not in payload["message"]
