"""Self-RAG inference loop as a LangGraph graph: step -> (step | finalize | rewrite | fallback)."""
import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Protocol, Sequence, TypedDict

from langgraph.graph import END as GRAPH_END
from langgraph.graph import StateGraph

from selfrag.config import Settings
from selfrag.config.overrides import ConfigOverrides, OverridesNotAllowed, apply_overrides
from selfrag.critics import critics
from selfrag.critics.critic import Critic
from selfrag.generation.citations import render_with_citations
from selfrag.generation.segment import generate
from selfrag.models import AnswerResponse, AnswerSegment, Beam, Candidate, Chunk, Filter, Usage
from selfrag.orchestrator import debug
from selfrag.orchestrator.streaming import ErrorEvent, StreamEvent, sync_leader
from selfrag.providers.interfaces import LLMClient
from selfrag.providers.usage import UsageCounter, track_usage
from selfrag.retrieval.multi_query import make_paraphrases
from selfrag.scoring.scores import is_end, s_rel, s_sup, s_use, segment_score, select_beams

log = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "I couldn't find this in the knowledge base."

REWRITE_PROMPT = """The search below found no relevant passages in the knowledge base.
Rewrite the question so a search engine is more likely to find the answer: use the
key terms and likely synonyms, and keep its meaning. Reply with the new question only.
Question: {question}"""


class InvalidOverride(ValueError):
    """An override that is in range but cannot be used here (the API maps it to 400)."""


class Retriever(Protocol):
    async def retrieve(self, query: str, filters: Filter, top_n: int,
                       variants: Sequence[str] = ()) -> list[tuple[Chunk, float]]: ...


@dataclass(frozen=True)
class SelfRAGConfig:
    delta: float = 0.2
    k: int = 5
    B: int = 2
    max_segments: int = 8
    w_rel: float = 1.0
    w_sup: float = 1.0
    w_use: float = 0.5
    hard_sup: bool = False
    min_rel: float = 0.2
    max_rewrites: int = 1
    force_retrieve_first: bool = True
    segment_unit: str = "answer"            # "sentence" | "answer"
    multi_query_variants: int = 2
    temperature: float = 0.0
    critic_mode: str = "verbal"             # logprobs | sample | verbal

    @classmethod
    def from_settings(cls, s: Settings) -> "SelfRAGConfig":
        return cls(delta=s.retrieve_threshold_delta, k=s.passages_per_step_k, B=s.beam_width_b,
                   max_segments=s.max_segments, w_rel=s.w_is_rel, w_sup=s.w_is_sup,
                   w_use=s.w_is_use, hard_sup=s.hard_constraint_no_support,
                   min_rel=s.min_rel_to_generate, max_rewrites=s.max_query_rewrites,
                   force_retrieve_first=s.force_retrieve_first, segment_unit=s.segment_unit,
                   multi_query_variants=s.multi_query_variants,
                   temperature=s.generator_temperature, critic_mode=s.critic_prob_mode)

    @property
    def weights(self) -> dict:
        return dict(w_rel=self.w_rel, w_sup=self.w_sup, w_use=self.w_use, hard_sup=self.hard_sup)


def new_trace_id() -> str:
    return f"tr_{uuid.uuid4().hex}"


class SelfRAGState(TypedDict, total=False):
    trace_id: str
    config: SelfRAGConfig       # frozen per-request snapshot: deployment defaults + overrides
    question: str
    filters: Filter
    beams: list[Beam]
    t: int                      # segment index
    rewrites: int
    answer: str
    best: Beam | None
    status: str                 # "ok" | "not_found"


class SelfRAG:
    def __init__(self, critic: Critic, generator: LLMClient, fast_llm: LLMClient,
                 retriever: Retriever, cfg: SelfRAGConfig = SelfRAGConfig()):
        if cfg.segment_unit not in ("sentence", "answer"):
            raise ValueError("segment_unit must be 'sentence' or 'answer'")
        self.critic, self.generator, self.fast_llm = critic, generator, fast_llm
        self.retriever, self.cfg = retriever, cfg
        self._critics = {critic.mode: critic}
        self.graph = self._build()

    def critic_for(self, cfg: SelfRAGConfig) -> Critic:
        if cfg.critic_mode not in self._critics:
            self._critics[cfg.critic_mode] = Critic(self.critic.llm, cfg.critic_mode, self.critic.n)
        return self._critics[cfg.critic_mode]

    def resolve_config(self, overrides: ConfigOverrides | None = None, *,
                       is_admin: bool = False) -> SelfRAGConfig:
        """Deployment config with the request's overrides applied (admin only)."""
        cfg = apply_overrides(self.cfg, overrides, is_admin=is_admin)
        try:
            self.critic_for(cfg)
        except ValueError as exc:                   # e.g. logprobs on an LLM without them
            raise InvalidOverride(str(exc)) from exc
        return cfg

    def _initial_state(self, question: str, filters: Filter, trace_id: str | None,
                       cfg: SelfRAGConfig | None = None) -> SelfRAGState:
        return {"question": question, "filters": filters, "beams": [Beam()], "t": 0,
                "rewrites": 0, "trace_id": trace_id or new_trace_id(), "config": cfg or self.cfg}

    @staticmethod
    def _package(state: SelfRAGState, usage: UsageCounter, start: float,
                 trace: debug.DebugTrace | None = None) -> AnswerResponse:
        if state["status"] == "ok":
            answer, segments = render_with_citations(state["best"])
        else:
            answer, segments = state["answer"], []
        return AnswerResponse(
            status=state["status"], answer=answer, segments=segments, trace_id=state["trace_id"],
            usage=Usage(llm_calls=usage.llm_calls, input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        latency_ms=round((time.perf_counter() - start) * 1000)),
            debug=trace.as_dict() if trace else None)

    async def answer(self, question: str, filters: Filter, trace_id: str | None = None,
                     cfg: SelfRAGConfig | None = None) -> SelfRAGState:
        return await self.graph.ainvoke(self._initial_state(question, filters, trace_id, cfg))

    @staticmethod
    def check_debug(want_debug: bool, is_admin: bool) -> None:
        if want_debug and not is_admin:
            raise OverridesNotAllowed("debug output is admin only")

    async def respond(self, question: str, filters: Filter,
                      overrides: ConfigOverrides | None = None, *,
                      is_admin: bool = False, debug_trace: bool = False) -> AnswerResponse:
        """Answer a question and package it with per-segment citations, status and usage.

        With debug_trace (admin only) the response also carries the decision trace."""
        self.check_debug(debug_trace, is_admin)
        cfg = self.resolve_config(overrides, is_admin=is_admin)
        start = time.perf_counter()
        with track_usage() as usage, debug.track_debug(debug_trace) as trace:
            if trace:
                trace.config = asdict(cfg)
            state = await self.answer(question, filters, cfg=cfg)
        return self._package(state, usage, start, trace)

    async def stream(self, question: str, filters: Filter,
                     overrides: ConfigOverrides | None = None, *,
                     is_admin: bool = False,
                     debug_trace: bool = False) -> AsyncIterator[StreamEvent]:
        """Like respond(), but yields each segment as soon as its beam leads after a step,
        a beam_switch when the leader changes, and finally done (or error).

        Invalid or forbidden overrides raise before any event is sent."""
        self.check_debug(debug_trace, is_admin)
        cfg = self.resolve_config(overrides, is_admin=is_admin)
        start = time.perf_counter()
        state = self._initial_state(question, filters, None, cfg)
        sent: list[AnswerSegment] = []
        with track_usage() as usage, debug.track_debug(debug_trace) as trace:
            if trace:
                trace.config = asdict(cfg)
            try:
                async for mode, chunk in self.graph.astream(state,
                                                            stream_mode=["updates", "values"]):
                    if mode == "values":
                        state = chunk
                    elif "step" in chunk:
                        beams = chunk["step"]["beams"]
                        if beams and beams[0].segments:     # beams are sorted, leader first
                            for event in sync_leader(sent, render_with_citations(beams[0])[1]):
                                yield event
            except Exception:
                log.exception("stream failed (trace %s)", state["trace_id"])
                yield StreamEvent("error", ErrorEvent(trace_id=state["trace_id"],
                                                      message="Answer generation failed."))
                return
            response = self._package(state, usage, start, trace)
        for event in sync_leader(sent, response.segments):    # finalize may pick another beam
            yield event
        yield StreamEvent("done", response)

    # ------------------------------------------------------------ one beam, one step

    async def decide_retrieval(self, question: str, beam: Beam, t: int,
                               cfg: SelfRAGConfig, log: dict | None = None) -> str:
        if t == 0 and cfg.force_retrieve_first:
            return "retrieve"
        p = await critics.retrieve(self.critic_for(cfg), question, " ".join(beam.segments),
                                   beam.last_passage.text if beam.last_passage else "")
        if log is not None:
            log["retrieve_probs"] = debug.probs(p)
        if beam.last_passage and p["Continue"] >= max(p["Yes"], p["No"]):
            return "continue"
        yes_ratio = p["Yes"] / (p["Yes"] + p["No"] + 1e-9)
        return "retrieve" if yes_ratio > cfg.delta else "no_retrieve"

    async def retrieve_for_segment(self, question: str, beam: Beam, filters: Filter,
                                   cfg: SelfRAGConfig, log: dict | None = None) -> list[Chunk]:
        prev = beam.segments[-1] if beam.segments else ""
        query = f"{question} {prev}".strip()            # paper: R given (x, y_{t-1})
        variants = await make_paraphrases(query, cfg.multi_query_variants, self.fast_llm)
        ranked = await self.retriever.retrieve(query, filters, top_n=cfg.k, variants=variants)
        if log is not None:
            log.update(query=query, variants=list(variants),
                       rerank_scores={c.chunk_id: round(float(s), 4) for c, s in ranked})
        return [c for c, _ in ranked]

    async def advance_beam(self, beam: Beam, question: str, t: int, filters: Filter,
                           cfg: SelfRAGConfig) -> list[Candidate]:
        unit, prefix, critic = cfg.segment_unit, beam.segments, self.critic_for(cfg)
        trace = debug.current()
        log = {"t": t, "prefix": " ".join(prefix)} if trace is not None else None
        decision = await self.decide_retrieval(question, beam, t, cfg, log)
        if log is not None:
            log["decision"] = decision

        if decision == "no_retrieve":               # Algorithm 1, lines 9-10: IsUse only
            c = await generate(self.generator, question, prefix, None, 0, unit,
                               cfg.temperature)
            if not is_end(c):
                c.use = await critics.isuse(critic, question, " ".join(prefix + [c.text]))
            self._log_step(trace, log, [], [], [c], cfg)
            return [c]

        passages = ([beam.last_passage] if decision == "continue"
                    else await self.retrieve_for_segment(question, beam, filters, cfg, log))
        # IsRel depends only on (x, y_<t, d): score before generating, skip irrelevant passages.
        rels = await asyncio.gather(*[critics.isrel(critic, question, " ".join(prefix), p.text)
                                      for p in passages])
        keep = [(p, r) for p, r in zip(passages, rels) if s_rel(r) >= cfg.min_rel]
        cands = await asyncio.gather(*[
            generate(self.generator, question, prefix, p, i + 1, unit, cfg.temperature)
            for i, (p, _) in enumerate(keep)])
        for c, (_, r) in zip(cands, keep):
            c.rel = r
        to_check = [c for c in cands if not is_end(c)]
        sups = await asyncio.gather(*[critics.issup(critic, question, c.passage.text, c.text)
                                      for c in to_check])
        for c, s in zip(to_check, sups):
            c.sup = s
        self._log_step(trace, log, passages, rels, cands, cfg)
        return list(cands)

    @staticmethod
    def _log_step(trace: debug.DebugTrace | None, log: dict | None, passages: list[Chunk],
                  rels: list[dict], cands: list[Candidate], cfg: SelfRAGConfig) -> None:
        """Add one beam's step to the debug trace: every passage and every candidate."""
        if trace is None or log is None:
            return
        scores = log.pop("rerank_scores", {})
        log["passages"] = [
            {**debug.passage_info(p), "rerank_score": scores.get(p.chunk_id),
             "rel_probs": debug.probs(r), "s_rel": round(s_rel(r), 3),
             "kept": s_rel(r) >= cfg.min_rel}
            for p, r in zip(passages, rels)]
        log["candidates"] = [
            {"text": c.text, "end": is_end(c),
             "chunk_id": c.passage.chunk_id if c.passage else None,
             "page": c.passage.page if c.passage else None,
             "s_rel": round(s_rel(c.rel), 3) if c.rel else None,
             "sup_probs": debug.probs(c.sup),
             "s_sup": round(s_sup(c.sup), 3) if c.sup else None,
             "use_probs": debug.probs(c.use),
             "segment_score": None if is_end(c) else round(segment_score(c, **cfg.weights), 3)}
            for c in cands]
        trace.steps.append(log)

    # ------------------------------------------------------------ graph nodes

    async def step(self, state: SelfRAGState) -> SelfRAGState:
        cfg = self._cfg(state)
        live = [b for b in state["beams"] if not b.done]
        done = [b for b in state["beams"] if b.done]
        expansions = await asyncio.gather(*[
            self.advance_beam(b, state["question"], state["t"], state["filters"], cfg)
            for b in live])
        new = select_beams(live, expansions, B=cfg.B, **cfg.weights)
        if cfg.segment_unit == "answer":       # the whole answer is one segment
            for b in new:
                b.done = True
        beams = sorted(done + new, key=lambda b: b.score, reverse=True)[:cfg.B]
        return {"beams": beams, "t": state["t"] + 1}

    async def finalize(self, state: SelfRAGState) -> SelfRAGState:
        cfg = self._cfg(state)
        finished = [b for b in state["beams"] if b.segments]
        uses = await asyncio.gather(*[critics.isuse(self.critic_for(cfg), state["question"],
                                                    " ".join(b.segments)) for b in finished])
        # IsUse on the full answer (t = T) is added to the beam's mean segment score.
        final = [b.score + cfg.w_use * s_use(p) for b, p in zip(finished, uses)]
        chosen = max(range(len(finished)), key=final.__getitem__)
        best = finished[chosen]
        if (trace := debug.current()) is not None:
            trace.final = [
                {"answer": " ".join(b.segments),
                 "segment_scores": [round(x, 3) for x in b.scores],
                 "mean_segment_score": round(b.score, 3), "use_probs": debug.probs(p),
                 "s_use": round(s_use(p), 3), "final_score": round(f, 3), "chosen": i == chosen}
                for i, (b, p, f) in enumerate(zip(finished, uses, final))]
        return {"answer": " ".join(best.segments), "best": best, "status": "ok"}

    async def fallback(self, state: SelfRAGState) -> SelfRAGState:
        return {"answer": NOT_FOUND_ANSWER, "best": None, "status": "not_found"}

    async def rewrite(self, state: SelfRAGState) -> SelfRAGState:
        out = await self.fast_llm.complete(REWRITE_PROMPT.format(question=state["question"]),
                                           max_tokens=200)
        question = out.text.strip() or state["question"]
        if (trace := debug.current()) is not None:
            trace.rewrites.append(question)
        return {"question": question, "beams": [Beam()], "t": 0,
                "rewrites": state.get("rewrites", 0) + 1}

    def _cfg(self, state: SelfRAGState) -> SelfRAGConfig:
        return state.get("config") or self.cfg

    def route(self, state: SelfRAGState) -> str:
        cfg, beams = self._cfg(state), state["beams"]
        if not any(b.segments for b in beams):      # nothing relevant was found
            return "rewrite" if state.get("rewrites", 0) < cfg.max_rewrites else "fallback"
        if all(b.done for b in beams) or state["t"] >= cfg.max_segments:
            return "finalize"
        return "step"

    def _build(self):
        g = StateGraph(SelfRAGState)
        g.add_node("step", self.step)
        g.add_node("finalize", self.finalize)
        g.add_node("rewrite", self.rewrite)
        g.add_node("fallback", self.fallback)
        g.set_entry_point("step")
        g.add_conditional_edges("step", self.route, {"step": "step", "finalize": "finalize",
                                                     "rewrite": "rewrite", "fallback": "fallback"})
        g.add_edge("rewrite", "step")
        g.add_edge("finalize", GRAPH_END)
        g.add_edge("fallback", GRAPH_END)
        return g.compile()
