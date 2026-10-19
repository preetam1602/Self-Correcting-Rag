# Self-RAG Implementation Guide

Oct 14, 2026 · @Dev Sup

## Overview

This guide builds a self-correcting RAG on the ideas of **Self-RAG** ([Asai et al., 2023](https://arxiv.org/abs/2310.11511)) without training or fine-tuning any LLM. The paper's four reflection tokens become four prompted critics, and its scoring formulas, adaptive retrieval threshold and segment-level beam search are implemented in application code.

The stack is unchanged from the first guide: Python 3.11, LangGraph, Qdrant (dense) + BM25 (sparse) fused with RRF (k = 60), `BAAI/bge-reranker-v2-m3` as reranker, and any instruction-tuned LLM.

**What Self-RAG adds over the first guide**

| First guide (CRAG-style) | This guide (Self-RAG) |
| --- | --- |
| Always retrieves for every question | Retrieves on demand: a critic decides per segment whether retrieval is needed |
| One answer from all top chunks together | One candidate continuation per passage, generated in parallel |
| Yes/no graders after the full answer | Graded critique per segment: relevance, support (3 levels), utility (1–5) |
| Retry loop when a grader fails | Candidates ranked by a weighted critique score; best one kept (beam search) |
| Citations checked after generation | Each segment carries its source and its support label by construction |

**What we keep from the paper, and what we change**

- Kept: the four critique types and their label sets, the paper's critic instructions (Appendix D), the score formulas (Eq. 3–4, Appendix A.3), the retrieval threshold δ, segment-level beam search with width B = 2, default weights IsRel 1.0, IsSup 1.0, IsUse 0.5.
- Changed: the paper trains Llama-2 to emit reflection tokens itself. Here a prompted LLM (the critic) emits them as constrained labels, and label probabilities come from token log-probabilities when the API exposes them, or from sampling otherwise.
- Added: RRF hybrid retrieval and a cross-encoder reranker in place of the paper's single Contriever retriever, so the K passages each segment sees are better.

A "segment" is one sentence, as in the paper. For short answers (one or two sentences) the whole answer can be one segment, which makes the loop much cheaper.

## Architecture

&#91;embedded content: Self-RAG inference loop · one pass per segment, 3 critics inside\]

The answer is built one segment at a time; for each segment the Retrieve critic chooses one of three paths, and only the best two partial answers survive each pass. IsUse (the fourth critic) scores the no-evidence path and re-ranks the finished answers.

## Reflection tokens as prompted critics

Each of the paper's four reflection tokens becomes one critic call that must answer with exactly one label from a fixed set. What the rest of the pipeline uses is not the label alone but a probability for each label, because the paper's scores (next sections) are built from those probabilities.

| Token | Input | Labels (best in bold) | Used for |
| --- | --- | --- | --- |
| Retrieve | question x, text so far y\<t, last evidence | **Yes**, No, Continue | Decide whether to retrieve before the next segment |
| IsRel | x, passage d | **Relevant**, Irrelevant | Score each passage |
| IsSup | x, d, candidate segment y\_t | **Fully**, Partially, No support | Score whether the segment is backed by d |
| IsUse | x, final answer y | 1, 2, 3, 4, **5** | Score overall usefulness |

### Getting label probabilities without training

The paper reads the trained model's probability for each reflection token. With an off-the-shelf LLM, use whichever of these your provider supports, in order of preference:

1. **Token log-probabilities** (vLLM, OpenAI-compatible servers, OpenAI API): ask for a one-word label, read `top_logprobs` of the first output token, and normalise over the allowed labels.
2. **Self-consistency sampling** (any API, including Claude): sample the label n = 5 times at temperature 0.7 and use label frequencies as probabilities.
3. **Verbalised distribution** (cheapest, least calibrated): ask the model to return a probability per label in JSON.

```python
import math, json, asyncio
from collections import Counter

class Critic:
    """Turns any chat LLM into a reflection-token predictor."""
    def __init__(self, llm, mode="logprobs", n_samples=5):
        self.llm, self.mode, self.n = llm, mode, n_samples

    async def label_probs(self, prompt: str, labels: list[str]) -> dict[str, float]:
        if self.mode == "logprobs":
            out = await self.llm.ainvoke(prompt, max_tokens=1, logprobs=True, top_logprobs=20)
            top = {t.token.strip().lower(): t.logprob for t in out.top_logprobs[0]}
            raw = {l: math.exp(top.get(l.split()[0].lower(), -30.0)) for l in labels}
        else:  # "sample"
            outs = await asyncio.gather(*[self.llm.ainvoke(prompt, temperature=0.7)
                                          for _ in range(self.n)])
            votes = Counter(self._match(o.content, labels) for o in outs)
            raw = {l: votes.get(l, 0) + 0.1 for l in labels}      # +0.1 smoothing
        z = sum(raw.values())
        return {l: v / z for l, v in raw.items()}

    @staticmethod
    def _match(text: str, labels: list[str]) -> str:
        t = text.strip().lower()
        return next((l for l in labels if t.startswith(l.lower())), labels[-1])
```

Use distinct first words for labels (Fully / Partially / No) so the first output token identifies the label. Run the critic on a small, fast model; the generator can be a larger one.

### Critic prompts (adapted from the paper's Appendix D)

```python
RETRIEVE_PROMPT = """You will see a question, the answer written so far, and optionally evidence
already retrieved. Decide what the NEXT sentence of the answer needs:
- Yes: it needs new factual evidence from the knowledge base.
- No: it needs no factual grounding (opinion, common sense, formatting, a greeting).
- Continue: the evidence already retrieved is enough to write it.
Question: {question}
Answer so far: {prefix}
Evidence already retrieved: {evidence}
Reply with one word: Yes, No or Continue."""

ISREL_PROMPT = """Decide whether the evidence is relevant to the question and the answer so far,
i.e. it gives useful information for completing the task. Keyword overlap alone
does not make it relevant.
Question: {question}
Answer so far: {prefix}
Evidence: {passage}
Reply with one word: Relevant or Irrelevant."""

ISSUP_PROMPT = """Judge whether the output sentence is supported by the evidence. Use only the
evidence, not your own knowledge.
- Fully: all information in the sentence is supported by, or extracted from, the evidence.
- Partially: some of it is supported, but major information is not in the evidence.
- No: the sentence ignores, is unrelated to, or contradicts the evidence.
Question: {question}
Evidence: {passage}
Output sentence: {segment}
Reply with one word: Fully, Partially or No."""

ISUSE_PROMPT = """Rate how helpful and informative the response is as an answer to the question,
regardless of whether it is factually correct.
5 complete and highly informative; 4 mostly fulfils the need, minor gaps;
3 acceptable but needs major additions; 2 addresses the request but is incomplete
or not relevant; 1 barely on-topic.
Question: {question}
Response: {answer}
Reply with one digit from 1 to 5."""
```

The paper uses the same definitions to collect labels from GPT-4 and reports 90–95% agreement with human judges for relevance, support and retrieval need, and about 80% for usefulness. Expect similar quality from a strong prompted critic, and lower from small models: check it on your own labelled sample (see Tuning).

## Adaptive retrieval

Before each segment, the Retrieve critic decides whether to search; the paper triggers retrieval when the normalised probability of Yes exceeds a threshold δ. When it fires, the query is the question plus the previous segment, and it runs through the unchanged hybrid retrieval → RRF → reranker pipeline from the first guide.

```latex
\frac{p(\text{Retrieve}=\text{Yes})}{p(\text{Retrieve}=\text{Yes}) + p(\text{Retrieve}=\text{No})} > \delta
```

The paper uses δ = 0.2 for most tasks and δ = 0 (always retrieve) where citations are mandatory. A lower δ retrieves more often; the paper's Figure 3c shows accuracy on PopQA falling sharply as δ rises, while PubHealth barely changes, so knowledge-heavy domains want a low δ.

Three outcomes, not two:

- **Yes above δ** → retrieve K = 5 new passages for this segment.
- **Continue** most likely → reuse the passages from the previous segment (no search cost).
- **Otherwise (No)** → generate the next segment without passages, then score it with IsUse only.

```python
async def decide_retrieval(state, critic, delta=0.2) -> str:
    if state["force_retrieve"]:
        return "retrieve"
    p = await critic.label_probs(RETRIEVE_PROMPT.format(
            question=state["question"],
            prefix=" ".join(state["segments"]),
            evidence=state["last_passage"].text if state.get("last_passage") else "none"),
        ["Yes", "No", "Continue"])
    if state.get("last_passage") and p["Continue"] >= max(p["Yes"], p["No"]):
        return "continue"
    yes_ratio = p["Yes"] / (p["Yes"] + p["No"] + 1e-9)
    return "retrieve" if yes_ratio > delta else "no_retrieve"

def retrieve_for_segment(state, ctx, k=5) -> list[Chunk]:
    prev = state["segments"][-1] if state["segments"] else ""
    query = f"{state['question']} {prev}".strip()           # paper: R given (x, y_{t-1})
    variants = [query] + make_paraphrases(query, n=2, llm=ctx.fast_llm)
    ids = hybrid_retrieve(variants, ctx)                    # dense + BM25 -> RRF (k=60), top 40
    ranked = ctx.reranker.rerank(query, ctx.store.get(ids), top_n=k, min_score=0.0)
    return [c for c, _ in ranked]
```

The reranker's `min_score` is set to 0 here on purpose: in Self-RAG, filtering bad passages is the job of the IsRel critic and the beam score, not of a hard cutoff. Keep the reranker for ordering and for choosing which K passages get a candidate.

**Cost lever:** the first segment almost always needs retrieval for a knowledge-base assistant, so set `force_retrieve=True` for t = 1 and skip that first critic call.

## Parallel candidate generation and critique

When retrieval fires, the generator writes one candidate next segment **per passage**, each seeing only that passage, and every candidate is critiqued for relevance and support. This is the core Self-RAG change: a bad passage produces a low-scoring candidate that is dropped, instead of polluting a single answer built from all passages at once.

```python
from dataclasses import dataclass

@dataclass
class Candidate:
    text: str                    # the new segment y_t
    passage: Chunk | None        # d, or None when no retrieval
    gen_logprob: float = 0.0     # mean token log-prob of y_t, if the API exposes it
    rel: dict | None = None      # IsRel label probabilities
    sup: dict | None = None      # IsSup label probabilities
    use: dict | None = None      # IsUse label probabilities

SEGMENT_PROMPT = """Write the NEXT single sentence of the answer to the question.
Use only the evidence below; if the evidence does not help, write a sentence that
needs no evidence, or reply exactly <END> if the answer is complete.
Question: {question}
Evidence [{n}] ({title}): {passage}
Answer so far: {prefix}
Next sentence:"""

async def gen_segment(state, passage, n, gen_llm) -> Candidate:
    prompt = SEGMENT_PROMPT.format(
        question=state["question"], n=n, title=passage.meta.get("title", ""),
        passage=passage.text, prefix=" ".join(state["segments"]) or "(none)")
    out = await gen_llm.ainvoke(prompt, temperature=0.0, max_tokens=120,
                                stop=["\n"], logprobs=True)
    lps = [t.logprob for t in (out.logprobs or [])]
    return Candidate(out.content.strip(), passage, sum(lps) / len(lps) if lps else 0.0)

async def critique(state, cand: Candidate, critic: Critic, score_use: bool):
    q, d = state["question"], cand.passage.text
    jobs = [critic.label_probs(ISREL_PROMPT.format(question=q,
                prefix=" ".join(state["segments"]), passage=d), ["Relevant", "Irrelevant"]),
            critic.label_probs(ISSUP_PROMPT.format(question=q, passage=d,
                segment=cand.text), ["Fully", "Partially", "No"])]
    if score_use:
        jobs.append(critic.label_probs(ISUSE_PROMPT.format(question=q,
                answer=" ".join(state["segments"] + [cand.text])), ["1", "2", "3", "4", "5"]))
    res = await asyncio.gather(*jobs)
    cand.rel, cand.sup = res[0], res[1]
    cand.use = res[2] if score_use else None
    return cand

async def expand(state, passages, gen_llm, critic, score_use=False) -> list[Candidate]:
    cands = await asyncio.gather(*[gen_segment(state, p, i + 1, gen_llm)
                                   for i, p in enumerate(passages)])
    return await asyncio.gather(*[critique(state, c, critic, score_use) for c in cands])
```

**Optimisation from the paper's design:** IsRel depends only on (x, d), so score it once per passage before generating, and skip generation for passages with p(Relevant) < 0.2. This cuts generator calls by roughly the share of irrelevant passages.

**No-retrieval branch:** generate one candidate with the question and prefix only, with no passage; it receives only an IsUse score (Algorithm 1, lines 9–10).

**Cost per retrieval step** with K = 5 passages and beam width B = 2: up to 10 generator calls and 20–30 critic calls, all parallel. With segment = whole answer (short-answer mode) this falls to 5 generator calls and about 15 critic calls per question.

## Critique scoring and segment beam search

Each candidate gets one score: its generation probability plus a weighted sum of its critique scores (paper Eq. 3–4). A beam of the best B = 2 partial answers is kept at every step, and the highest-scoring complete answer is returned.

```latex
f(y_t, d) = p(y_t \mid x, d, y_{<t}) + \sum_{G \in \{\text{IsRel},\,\text{IsSup},\,\text{IsUse}\}} w^G s_t^G
```

The per-critic scores, from the paper's Appendix A.3:

```latex
s^{\text{IsRel}} = \frac{p(\text{Relevant})}{p(\text{Relevant}) + p(\text{Irrelevant})}
```

```latex
s^{\text{IsSup}} = \frac{p(\text{Fully})}{S} + 0.5 \cdot \frac{p(\text{Partially})}{S}, \quad S = p(\text{Fully}) + p(\text{Partially}) + p(\text{No})
```

```latex
s^{\text{IsUse}} = \sum_{i=1}^{5} w_i \frac{p(\text{IsUse}=i)}{S}, \quad w = \{-1, -0.5, 0, 0.5, 1\}
```

Default weights from the paper: w(IsRel) = 1.0, w(IsSup) = 1.0, w(IsUse) = 0.5. Raising w(IsSup) increases citation precision but makes answers shorter and less fluent (paper Figure 3b), so raise it for compliance-sensitive use and lower it for open-ended writing.

```python
USE_W = {"1": -1.0, "2": -0.5, "3": 0.0, "4": 0.5, "5": 1.0}

def s_rel(p): return p["Relevant"] / (p["Relevant"] + p["Irrelevant"] + 1e-9)
def s_sup(p):
    z = p["Fully"] + p["Partially"] + p["No"] + 1e-9
    return p["Fully"] / z + 0.5 * p["Partially"] / z
def s_use(p):
    z = sum(p.values()) + 1e-9
    return sum(USE_W[k] * v / z for k, v in p.items())

def segment_score(c: Candidate, w_rel=1.0, w_sup=1.0, w_use=0.5,
                  hard_sup=False) -> float:
    if hard_sup and c.sup and max(c.sup, key=c.sup.get) == "No":
        return float("-inf")                       # paper's hard-constraint option
    score = math.exp(c.gen_logprob) if c.gen_logprob else 0.0   # p(y_t | x, d, y_<t)
    if c.rel: score += w_rel * s_rel(c.rel)
    if c.sup: score += w_sup * s_sup(c.sup)
    if c.use: score += w_use * s_use(c.use)
    return score
```

If your API gives no generation log-probabilities, the first term is 0 and ranking relies on the critics alone; that is fine in practice because the critique terms carry most of the signal.

### Segment-level beam search

```python
@dataclass
class Beam:
    segments: list[str]
    citations: list[Chunk | None]
    scores: list[float]
    last_passage: Chunk | None = None
    done: bool = False
    @property
    def score(self): return sum(self.scores) / max(len(self.scores), 1)

def select_beams(beams: list[Beam], expansions: list[list[Candidate]], B=2, **w) -> list[Beam]:
    pool = []
    for beam, cands in zip(beams, expansions):
        for c in cands:
            done = c.text == "<END>" or not c.text
            pool.append(Beam(beam.segments + ([] if done else [c.text]),
                             beam.citations + ([] if done else [c.passage]),
                             beam.scores + ([] if done else [segment_score(c, **w)]),
                             c.passage or beam.last_passage, done))
    return sorted(pool, key=lambda b: b.score, reverse=True)[:B]
```

The paper ranks by the segment score; averaging across segments (as above) is our choice to avoid favouring longer answers. Each kept segment is cited to its own passage, so citations come for free: render the answer as `segment [n]` using `beam.citations`.

## Orchestration (LangGraph)

The graph has three nodes: `step` advances every live beam by one segment (decide → retrieve → generate → critique → select), `finalize` re-ranks finished answers by IsUse, and `fallback` handles the case where no passage was relevant. `step` loops until all beams end or a segment cap is hit.

### State and one decoding step

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class SelfRAGState(TypedDict, total=False):
    question: str
    beams: list[Beam]
    t: int                       # segment index
    rewrites: int
    answer: str
    status: str                  # "ok" | "not_found"

CFG = dict(delta=0.2, k=5, B=2, max_segments=8, w_rel=1.0, w_sup=1.0, w_use=0.5,
           hard_sup=False, min_rel=0.2, max_rewrites=1)

async def advance_beam(beam: Beam, question: str, t: int) -> list[Candidate]:
    st = {"question": question, "segments": beam.segments,
          "last_passage": beam.last_passage, "force_retrieve": t == 0}
    decision = await decide_retrieval(st, critic, CFG["delta"])
    if decision == "no_retrieve":
        c = await gen_no_evidence(st, gen_llm)                  # prompt without passage
        c.use = await critic.label_probs(ISUSE_PROMPT.format(question=question,
                    answer=" ".join(beam.segments + [c.text])), list("12345"))
        return [c]
    passages = ([beam.last_passage] if decision == "continue"
                else retrieve_for_segment(st, ctx, k=CFG["k"]))
    rel = await asyncio.gather(*[critic.label_probs(ISREL_PROMPT.format(
              question=question, prefix=" ".join(beam.segments), passage=p.text),
              ["Relevant", "Irrelevant"]) for p in passages])
    keep = [(p, r) for p, r in zip(passages, rel) if s_rel(r) >= CFG["min_rel"]]
    cands = await asyncio.gather(*[gen_segment(st, p, i + 1, gen_llm)
                                   for i, (p, _) in enumerate(keep)])
    for c, (_, r) in zip(cands, keep):
        c.rel = r
        c.sup = await critic.label_probs(ISSUP_PROMPT.format(question=question,
                    passage=c.passage.text, segment=c.text), ["Fully", "Partially", "No"])
    return list(cands)

async def step(state: SelfRAGState) -> SelfRAGState:
    live = [b for b in state["beams"] if not b.done]
    done = [b for b in state["beams"] if b.done]
    exps = await asyncio.gather(*[advance_beam(b, state["question"], state["t"])
                                  for b in live])
    w = {k: CFG[k] for k in ("w_rel", "w_sup", "w_use", "hard_sup")}
    new = select_beams(live, exps, B=CFG["B"], **w) if any(exps) else []
    beams = sorted(done + new, key=lambda b: b.score, reverse=True)[:CFG["B"]]
    return {"beams": beams, "t": state["t"] + 1}
```

### Finalize, fallback and routing

```python
async def finalize(state: SelfRAGState) -> SelfRAGState:
    finished = [b for b in state["beams"] if b.segments]
    for b in finished:                                       # IsUse on the full answer (t = T)
        p = await critic.label_probs(ISUSE_PROMPT.format(question=state["question"],
                answer=" ".join(b.segments)), list("12345"))
        b.scores.append(CFG["w_use"] * s_use(p))
    best = max(finished, key=lambda b: b.score)
    return {"answer": render_with_citations(best), "status": "ok"}

def fallback(state: SelfRAGState) -> SelfRAGState:
    return {"status": "not_found",
            "answer": "I couldn't find this in the knowledge base."}

def route(state: SelfRAGState) -> str:
    beams = state["beams"]
    if not beams or not any(b.segments for b in beams):
        return "rewrite" if state.get("rewrites", 0) < CFG["max_rewrites"] else "fallback"
    if all(b.done for b in beams) or state["t"] >= CFG["max_segments"]:
        return "finalize"
    return "step"

def rewrite(state: SelfRAGState) -> SelfRAGState:            # reused from the first guide
    q = rewrite_query({"question": state["question"], "queries": [state["question"]],
                       "grader_feedback": "no relevant passages"}, fast_llm)
    return {"question": q, "beams": [Beam([], [], [])], "t": 0,
            "rewrites": state.get("rewrites", 0) + 1}

g = StateGraph(SelfRAGState)
g.add_node("step", step); g.add_node("finalize", finalize)
g.add_node("rewrite", rewrite); g.add_node("fallback", fallback)
g.set_entry_point("step")
g.add_conditional_edges("step", route, {"step": "step", "finalize": "finalize",
                                        "rewrite": "rewrite", "fallback": "fallback"})
g.add_edge("rewrite", "step")
g.add_edge("finalize", END); g.add_edge("fallback", END)
app = g.compile()

result = await app.ainvoke({"question": q, "beams": [Beam([], [], [])], "t": 0})
```

`gen_no_evidence` is `gen_segment` with the evidence line removed; `render_with_citations` joins `beam.segments` and appends `[n]` with each segment's passage title and URL. The query rewrite is not in the paper; it is kept from the first guide as a safety net for when every retrieved passage scores irrelevant.

## Tuning, evaluation & configuration

Without a trained model, the critic's accuracy is the main quality risk, so validate it before tuning anything else. Then tune δ and the IsSup weight, which the paper shows to be the two most influential inference-time controls.

### Configuration reference

```yaml
retrieval:                 # unchanged from the first guide
  dense_top_k: 50
  sparse_top_k: 50
  multi_query_variants: 2
  rrf_k: 60
  fused_top_n: 40
  reranker: BAAI/bge-reranker-v2-m3
self_rag:
  passages_per_step_k: 5      # paper: top 5 at inference
  retrieve_threshold_delta: 0.2   # paper default; 0 = always retrieve
  force_retrieve_first: true
  beam_width_B: 2             # paper default
  max_segments: 8
  segment_unit: sentence      # or "answer" for short-answer mode
  weights: { is_rel: 1.0, is_sup: 1.0, is_use: 0.5 }   # paper defaults
  hard_constraint_no_support: false
  min_rel_to_generate: 0.2
  max_query_rewrites: 1       # not in paper; safety net
critic:
  model: <small fast LLM>
  prob_mode: logprobs         # logprobs | sample | verbal
  n_samples: 5
generator:
  model: <main LLM>
  temperature: 0.0
```

### Validation and tuning order

1. **Critic agreement:** label 50 samples per critic by hand (the paper used 20 per aspect); target ≥ 85% agreement for IsRel and IsSup. If lower, use a stronger critic model or add 2–3 few-shot examples to its prompt.
2. **Retrieval recall@40** of hybrid + RRF, as in the first guide (target ≥ 0.90).
3. **δ sweep:** try 0, 0.2, 0.4, 0.6; plot accuracy against retrieval frequency and pick the knee.
4. **IsSup weight:** try 0.5, 1.0, 2.0; higher favours citation precision, lower favours fluency and completeness.
5. **Beam width and K:** B = 1 and K = 3 roughly halve cost; measure the quality drop before adopting.
6. **Ablation** on your golden set: first guide (CRAG-style) vs this guide, measured on correctness, citation precision, "not found" rate, p95 latency and cost per query. Keep Self-RAG where it wins on citation precision enough to justify its higher call count.

### What to log per request

Retrieve decision and yes-ratio per segment; passages per step with reranker score and IsRel probability; every candidate with its IsSup and IsUse probabilities and its final score; beams kept per step; final answer with per-segment citations. The share of segments with IsSup = No is the best single health signal.

### Sources

- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) — Asai et al., 2023 (the paper provided). Algorithm 1, Eq. 3–4, Appendix A.3 and Appendix D are the basis of this guide.
- Self-correcting RAG — Implementation Guide (the first guide): retrieval, RRF, reranker and query rewriter reused here.
