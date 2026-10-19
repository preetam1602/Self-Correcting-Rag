# Self-RAG Service — Software Design Plan

Oct 15, 2026 · @Dev Sup

## Goals, non-goals and requirements

We will ship the Self-RAG pipeline from the *Self-RAG Implementation Guide* as a standalone Python service: an ingestion worker plus a question-answering API that returns answers cited sentence by sentence. The design reuses the guide's stack unchanged and makes every Self-RAG parameter configurable without code changes.

**Goals**

- Answer questions over a private knowledge base, with every sentence cited to a source passage.
- Implement the guide faithfully: adaptive retrieval (δ), four prompted critics, one candidate per passage, weighted scoring, beam search (B = 2).
- Keep the model providers swappable: any LLM for generator and critic, any reranker behind one interface.
- Make quality measurable from day one: a golden set, per-stage metrics and full request traces.

**Non-goals (v1)**

- Training or fine-tuning any LLM (as agreed for the guide).
- Multi-turn chat memory; v1 answers single questions (conversation history can be passed as context later).
- Web search fallback; the knowledge base is the only source.
- A user-facing UI beyond a minimal test console.

**Functional requirements**

| ID | Requirement |
| --- | --- |
| F1 | Ingest PDF, HTML, DOCX and Markdown; re-index changed documents incrementally |
| F2 | Hybrid retrieval (dense + BM25) fused with RRF, reranked by a cross-encoder |
| F3 | Self-RAG inference loop with the four critics and beam search |
| F4 | Return answer, per-sentence citations, status (`ok` / `not_found`) and a trace ID |
| F5 | Filter retrieval by tenant and document ACLs |
| F6 | Stream the answer to the client as segments are accepted |
| F7 | Expose config (δ, K, B, weights, critic mode) per deployment and per request (admin only) |

**Non-functional targets (to confirm with product)**

| Metric | Target |
| --- | --- |
| p95 latency, short-answer mode | ≤ 6 s |
| p95 latency, sentence mode (≤ 8 segments) | ≤ 20 s, first segment ≤ 5 s |
| Faithfulness (IsSup = Fully on golden set) | ≥ 0.90 |
| Availability | 99.5% monthly |
| Corpus size, v1 | up to 1M chunks |

## System architecture

&#91;embedded content: service architecture · 2 paths, 4 model endpoints, 5 stores\]

The answer service and the ingestion workers are separate deployments that share the same stores, so heavy uploads never slow down answers. The orchestrator is the only component that knows the Self-RAG control flow; every other module is a stateless step it calls, and the answer service also uses the embedder for query embeddings.

## Code structure

One Python monorepo, one package per pipeline stage, with all model providers behind interfaces in `providers/`. Stage packages depend only on interfaces, never on a vendor SDK, so swapping Qdrant, the reranker or the LLM is a config change.

```
selfrag/
├── api/            FastAPI app: routes, auth, SSE streaming, request validation
├── config/         Pydantic settings; YAML defaults + env overrides + per-request overrides
├── ingestion/      parsers, chunker, enricher, embed + index jobs, change detection
├── retrieval/      dense_search, sparse_search, rrf, multi_query, hybrid_retrieve
├── rerank/         Reranker interface + cross-encoder and API implementations
├── critics/        Critic (label_probs), prompts, the four critic wrappers
├── generation/     segment generator, no-evidence generator, citation renderer
├── scoring/        s_rel, s_sup, s_use, segment_score, Beam, select_beams
├── orchestrator/   LangGraph graph: step, finalize, rewrite, fallback, routing
├── providers/      LLMClient, Embedder, VectorStore, SparseIndex, DocStore interfaces + adapters
├── observability/  tracing (OpenTelemetry/Langfuse), metrics, structured logs
├── eval/           golden-set runner, critic-agreement tool, ablation harness
└── tests/          unit, contract, integration, regression (golden set)
```

| Package | Maps to guide section | Owns | Depends on |
| --- | --- | --- | --- |
| `retrieval` | Adaptive retrieval (first guide: RRF) | Hybrid search + RRF fusion | `providers` |
| `rerank` | Adaptive retrieval | Ordering and top-K choice | `providers` |
| `critics` | Reflection tokens as prompted critics | Label probabilities for 4 tokens | `providers` (LLM) |
| `generation` | Parallel candidate generation | One candidate per passage | `providers` (LLM) |
| `scoring` | Critique scoring and beam search | Paper formulas, beams | none (pure functions) |
| `orchestrator` | Orchestration (LangGraph) | Control flow, caps, fallback | all stage packages |

**Design rules**

- `scoring` stays pure (no I/O): it is where the paper's maths lives, so it gets exhaustive unit tests.
- Every network call is `async` and goes through one `providers` adapter that handles timeouts, retries with backoff, and token/cost accounting.
- Prompts live as versioned files in `critics/prompts/` and `generation/prompts/`; the prompt version is written into every trace.

## Core interfaces and data models

Six provider interfaces isolate every external dependency, and four data models carry state through the pipeline. Everything in the guide's code maps onto these types.

### Provider interfaces

```python
from typing import Protocol, Sequence

class LLMClient(Protocol):
    async def complete(self, prompt: str, *, max_tokens: int, temperature: float = 0.0,
                       stop: list[str] | None = None, logprobs: bool = False,
                       top_logprobs: int = 0) -> "Completion": ...
    @property
    def supports_logprobs(self) -> bool: ...          # decides the critic's prob mode

class Embedder(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...

class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence["Chunk"], vectors: Sequence[list[float]]) -> None: ...
    async def search(self, vector: list[float], k: int, filters: "Filter") -> list[str]: ...
    async def delete_by_doc(self, doc_id: str) -> None: ...

class SparseIndex(Protocol):                         # BM25: OpenSearch or Qdrant sparse
    async def upsert(self, chunks: Sequence["Chunk"]) -> None: ...
    async def search(self, query: str, k: int, filters: "Filter") -> list[str]: ...

class DocStore(Protocol):                            # chunk text + metadata by chunk_id
    async def get_many(self, ids: Sequence[str]) -> list["Chunk"]: ...

class Reranker(Protocol):
    async def rerank(self, query: str, chunks: Sequence["Chunk"], top_n: int) -> list[tuple["Chunk", float]]: ...
```

### Data models

| Model | Key fields | Lifetime |
| --- | --- | --- |
| `Chunk` | `chunk_id`, `doc_id`, `text`, `title`, `section`, `page`, `source_url`, `tenant_id`, `acl`, `updated_at` | Stored at ingestion |
| `Candidate` | `text`, `passage`, `gen_logprob`, `rel`, `sup`, `use` (label → probability) | One decoding step |
| `Beam` | `segments`, `citations`, `scores`, `last_passage`, `done` | One request |
| `SelfRAGState` | `question`, `beams`, `t`, `rewrites`, `answer`, `status`, `trace_id`, `config` | One request (LangGraph state) |
| `AnswerResponse` | `answer`, `segments[{text, citation}]`, `status`, `trace_id`, `usage` | Returned to client |

`Candidate` and `Beam` are exactly the dataclasses from the implementation guide; `SelfRAGState` adds `trace_id` and a frozen `config` snapshot so a request's behaviour is reproducible from its trace.

## API design

A small REST API with one main endpoint, `POST /v1/answer`, offered in blocking and streaming (Server-Sent Events) forms. Streaming sends each segment only after its beam is the leader at the end of a step, so clients never see text that is later discarded except when the leading beam changes, which is signalled explicitly.

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| POST | `/v1/answer` | Answer a question (blocking) | API key / JWT with tenant claim |
| POST | `/v1/answer/stream` | Same, as SSE events | same |
| POST | `/v1/documents` | Upload or register documents for ingestion | admin |
| DELETE | `/v1/documents/{doc_id}` | Remove a document from all indexes | admin |
| GET | `/v1/jobs/{job_id}` | Ingestion job status | admin |
| GET | `/v1/traces/{trace_id}` | Full decision trace for a request | admin |
| GET | `/healthz`, `/readyz` | Liveness, readiness (stores + model endpoints reachable) | none |

**Request and response**

```json
// POST /v1/answer
{
  "question": "What is the notice period for contractors?",
  "mode": "sentence",                 // "sentence" | "answer" (short-answer mode)
  "filters": { "collection": "hr-policies" },
  "overrides": { "delta": 0.1 }       // admin only; validated against allowed ranges
}

// 200 OK
{
  "status": "ok",                     // "ok" | "not_found" | "timeout"
  "answer": "Contractors must give 30 days' notice [1]. ...",
  "segments": [
    { "text": "Contractors must give 30 days' notice.",
      "citation": { "n": 1, "doc_id": "d_812", "title": "Contractor Policy",
                    "page": 4, "url": "https://...", "support": "fully" } }
  ],
  "trace_id": "tr_01J...",
  "usage": { "llm_calls": 23, "input_tokens": 18400, "output_tokens": 610, "latency_ms": 7420 }
}
```

**SSE events:** `segment` (text + citation), `beam_switch` (leading beam changed; client replaces text from index i), `done` (final payload as above), `error`.

**Errors:** 400 invalid request or override out of range; 401/403 auth; 429 tenant rate limit; 503 model endpoint or store unavailable after retries; 504 request exceeded its time budget (partial answer returned with `status: "timeout"` when at least one segment was accepted).

## Data and storage

Three stores hold the corpus under one shared `chunk_id`, and Redis holds caches only, so it can be flushed at any time. Ingestion is an asynchronous job queue so uploads never block the answer path.

| Store | Holds | Technology | Notes |
| --- | --- | --- | --- |
| Vector index | Chunk embeddings + filter fields | Qdrant (or pgvector) | Payload index on `tenant_id`, `collection`, `acl` |
| Sparse index | BM25 terms | OpenSearch, or Qdrant sparse vectors | Same filter fields as vector index |
| Doc store | Chunk text + full metadata, document versions | PostgreSQL | Source of truth; indexes can be rebuilt from it |
| Object storage | Original files | S3-compatible | Linked from `Chunk.source_url` |
| Cache | Query embeddings, reranker scores, critic label probs | Redis, TTL 24 h | Key = hash(model, prompt version, inputs) |
| Traces | Per-request decisions and scores | Langfuse / OTel backend | Retained 30 days (to confirm) |

### Ingestion pipeline

1. `POST /v1/documents` stores the file in object storage and enqueues a job (Celery or Arq on Redis).
2. Worker parses, chunks (400–800 tokens, 10–15% overlap) and enriches with metadata, as in the first guide.
3. Change detection: a content hash per chunk; unchanged chunks are skipped, removed chunks are deleted from all three stores.
4. Embed in batches, then write doc store first, then vector and sparse indexes; the document becomes searchable only after all three succeed (status flag in PostgreSQL).
5. A nightly consistency job compares chunk counts across stores and repairs drift.

**Re-embedding:** a model change creates a new Qdrant collection built in the background, then switches an alias; no downtime and instant rollback.

**Critic cache** is the largest cost saver: IsRel depends only on (question, passage), so repeated or similar questions reuse it. Cache keys include the prompt version so a prompt change never serves stale labels.

## Non-functional design

Latency and cost are driven by LLM call count, so the design bounds calls per request with hard caps, runs every independent call in parallel, and degrades gracefully instead of failing.

### Latency and cost control

- **Parallelism:** the K candidates, their critics and both beams run concurrently (`asyncio.gather`); a step's latency is one generator call plus one critic round, not K of them.
- **Hard caps per request:** `max_segments` = 8, max 60 LLM calls, max time budget 25 s; on hitting a cap the orchestrator jumps to `finalize` with the best beam so far.
- **Cheap paths first:** `force_retrieve_first` skips the first Retrieve critic; IsRel is computed before generation and passages below 0.2 are never expanded; critic results are cached.
- **Short-answer mode** (one segment = whole answer) as the default for FAQ-style traffic; sentence mode for long-form questions, chosen by the client or by a length heuristic.
- **Model split:** critic on a small fast model, generator on the main model; reranker on GPU.

### Scaling

| Component | Scales by | Notes |
| --- | --- | --- |
| API + orchestrator | Stateless pods, horizontal autoscale on in-flight requests | Long-lived SSE connections: autoscale on concurrency, not CPU |
| Reranker | GPU deployment (Triton or TEI), batch size 32 | One GPU handles roughly tens of requests/s at 40 pairs each (to benchmark) |
| LLM calls | Provider rate limits; self-hosted vLLM if logprobs are needed | Per-tenant token budgets enforced at the adapter |
| Ingestion workers | Queue depth | Isolated from the answer path |
| Qdrant / OpenSearch | Shards and replicas | Sized for 1M chunks in v1 |

### Reliability

- Timeouts on every provider call (LLM 20 s, reranker 3 s, stores 2 s) with 2 retries and jittered backoff.
- Circuit breaker per provider; if the critic model is down, fall back to a degraded mode (reranker-score ranking only, flagged `degraded: true` in the response).
- If the sparse index is down, retrieval continues dense-only (RRF over one list).

### Security

- Tenant and ACL filters are applied **inside** the vector and sparse queries, never after retrieval, so unauthorised chunks never reach the LLM.
- Retrieved text is wrapped as quoted evidence in every prompt, and prompts tell the model to ignore instructions inside it (prompt-injection defence); the IsSup critic adds a second check.
- PII: traces store chunk IDs, not chunk text, by default; question text retention is configurable per tenant.
- Secrets via the platform secret manager; no provider keys in config files.

## Observability, evaluation and testing

Every request produces one trace with a span per pipeline step, and a golden-set regression gate blocks any merge that changes prompts, models or config defaults without passing it.

### Tracing and metrics

- **Spans:** `retrieve_decision`, `hybrid_retrieve`, `rrf`, `rerank`, `isrel`, `generate`, `issup`, `isuse`, `select_beams`, `finalize`, each with inputs (IDs), label probabilities, scores, tokens and latency.
- **Dashboards:** p50/p95 latency by mode; LLM calls and cost per request; retrieval rate (share of segments with Retrieve = Yes); share of segments with IsSup = No; `not_found` rate; cache hit rate; provider error rate.
- **Alerts:** p95 above target for 15 min; IsSup = No share up more than 50% week on week; `not_found` rate above 15%; any provider circuit open.

### Evaluation

| Suite | Size | Measures | When |
| --- | --- | --- | --- |
| Critic agreement | 50 labelled items per critic | Agreement with human labels (target ≥ 85% IsRel, IsSup) | Any critic prompt or model change |
| Retrieval | 150–300 golden questions | Recall@40 after RRF (≥ 0.90), nDCG@5 after rerank | Chunking, embedder, reranker change |
| End-to-end | Same golden set | Correctness, faithfulness, citation precision, `not_found` rate, latency, cost | Every release; nightly |
| Ablation | Same golden set | First guide (CRAG-style) vs Self-RAG; δ and IsSup-weight sweeps | Once in Phase 3, then quarterly |

### Testing strategy

- **Unit:** `scoring` formulas against hand-computed values (e.g. IsSup Fully 0.6 / Partially 0.3 / No 0.1 → 0.75); RRF ordering; beam selection and tie-breaking; routing functions for every branch.
- **Contract:** each provider adapter against a recorded fixture; critics must parse every label variant and fall back safely on malformed output.
- **Integration:** full graph against a 200-document test corpus with a deterministic fake LLM, covering retrieve / continue / no-retrieve, all-irrelevant → rewrite → fallback, cap hits and timeouts.
- **Regression gate:** end-to-end golden set in CI on labelled PRs; fail if faithfulness or correctness drops more than 2 points.
- **Load:** k6 or Locust at 2× expected peak concurrency, measuring p95 and provider rate-limit headroom.

## Delivery plan

&#91;embedded content: delivery roadmap · 5 phases, 4 gates, \~12 weeks\]

The plan assumes a team of 2–3 engineers; the week counts are estimates to confirm once providers are chosen. Gate 2 comes before the full loop is built on purpose: if the prompted critic cannot reach 85% agreement, the fix (a stronger critic model or a self-hosted one with logprobs) changes the cost model for everything after it.

## Risks and open questions

The biggest risk is that a prompted critic is less accurate than the paper's trained one; the plan puts its validation gate before any tuning or launch.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Prompted critic disagrees with humans | Wrong passages kept, unsupported sentences pass | Gate 1 (≥ 85% agreement); stronger critic model; few-shot prompts |
| No token logprobs from chosen LLM (e.g. Claude API) | Sampling mode multiplies critic calls by 5 | Self-host a critic on vLLM, or use verbalised mode for low-stakes critics (IsUse) |
| Sentence mode too slow or costly | Misses latency target | Short-answer mode default; caps; caching; B = 1 / K = 3 fallback config |
| Beam switches confuse streaming clients | Visible text rewrites | Stream only after a step settles; `beam_switch` event; option to stream only at `done` |
| Corpus gaps read as model failures | High `not_found` rate | Log unanswered questions per collection; weekly content-gap report |
| Index drift between stores | Missing or orphan chunks | Doc store as source of truth; nightly consistency job |

**Open questions**

- [ ] Which LLM provider(s) for generator and critic, and do they expose logprobs?
- [ ] Expected peak QPS and corpus size at launch (drives Qdrant sizing and GPU count)?
- [ ] Is sentence mode needed in v1, or is short-answer mode enough for launch?
- [ ] Trace and question-text retention period per tenant?
- [ ] Who labels the golden set and critic-agreement samples, and by when?
