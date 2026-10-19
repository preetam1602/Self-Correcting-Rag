# Self-RAG Service

A citation-grounded question-answering service for private knowledge bases, implementing [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511) with prompted critics instead of fine-tuned models.

**Main goal: run fully on local LLMs** via [Ollama](https://ollama.com), so documents and models stay on your machine. This needs a GPU. **On a CPU-only system, a cloud API (Groq or Gemini) is required.** Self-RAG makes many LLM calls per question, and a local model on a CPU takes minutes per answer. Embeddings, reranking and the databases always run locally.

## Features

- Hybrid retrieval (dense + BM25, fused with RRF) and cross-encoder reranking
- Four reflection critics (Retrieve, IsRel, IsSup, IsUse) with beam search
- Answers with source citations, plus SSE streaming
- Incremental ingestion and tenant / collection / ACL filtering

**Stack:** Python 3.11+, Qdrant, PostgreSQL, LangGraph, `BAAI/bge-m3` embeddings, `BAAI/bge-reranker-v2-m3` reranker.

## Setup

```bash
docker compose up -d                 # Qdrant + PostgreSQL
python -m venv .venv
.venv\Scripts\Activate.ps1           # Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Create a `.env` in the project root (never commit it):

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=selfrag
POSTGRES_PASSWORD=your_db_password
DATABASE_URL=your_database_url
CRITIC_PROB_MODE=verbal
```

Then add **one** of the following LLM setups.

**A. Local LLM (GPU).** First download a model yourself (`ollama pull qwen3:4b`). selfrag never pulls models.

```dotenv
LLM_PROVIDER=ollama
GENERATOR_MODEL=qwen3:4b
CRITIC_MODEL=qwen3:4b        # same model = only one copy in memory
```

**B. Cloud API (required on CPU-only systems)**

```dotenv
LLM_PROVIDER=groq            # or gemini, with GEMINI_API_KEY
GROQ_API_KEY=your_api_key
GENERATOR_MODEL=openai/gpt-oss-20b
CRITIC_MODEL=openai/gpt-oss-20b
```

## Usage

```bash
python -m selfrag.ingestion ./documents --tenant default --collection hr-policies --acl hr-team,execs
python -m pytest
```

Ingestion accepts PDF, HTML, DOCX and Markdown files. To ask questions, call `await rag.respond(question, reader)` from Python. It returns the answer, its citations, a status, a trace ID and usage figures.

All settings live in `selfrag/config/defaults.yaml`, and environment variables override them. That file includes the CPU-safe Ollama limits: threads, concurrency, context size and the idle unload time.

## Status and limitations

- **Done:** ingestion, retrieval, reranking, critics, beam search, citations, access filtering, streaming, local and cloud LLM support.
- **Pending:** HTTP API, auth, background ingestion, traces/metrics, evaluation.
- **On CPU:** the reranker takes about 15 s per passage and the first indexing run is slow. Local LLMs on a CPU alone are too slow, so use option B.
- **Calibration:** `verbal` critic scores still need evaluation against labelled examples.

See `Self-RAG Implementation Guide.md` and `Self-RAG Service — Software Design Plan.md` for the full design.
