"""Long-lived components the API shares across requests, built once at startup."""
import asyncio
from dataclasses import dataclass, replace

import httpx
from qdrant_client import AsyncQdrantClient

from selfrag.config import Settings
from selfrag.critics.critic import Critic
from selfrag.ingestion.pipeline import ChunkingConfig, IngestionPipeline
from selfrag.orchestrator.graph import SelfRAG, SelfRAGConfig
from selfrag.providers.embedder import SentenceTransformerEmbedder
from selfrag.providers.gemini import GeminiClient
from selfrag.providers.groq import GroqClient
from selfrag.providers.ollama import OllamaClient
from selfrag.providers.postgres import PostgresDocStore
from selfrag.providers.qdrant import QdrantSparseIndex, QdrantVectorStore
from selfrag.rerank.cross_encoder import CrossEncoderReranker
from selfrag.retrieval.hybrid_retrieve import HybridRetriever

MODES = ("sentence", "answer")


@dataclass
class Services:
    rags: dict[str, SelfRAG]            # one per segment unit; they share every component
    default_mode: str
    pipeline: IngestionPipeline
    client: AsyncQdrantClient
    vector_store: QdrantVectorStore
    sparse_index: QdrantSparseIndex
    doc_store: PostgresDocStore

    async def ready(self) -> dict[str, bool]:
        async def check(probe) -> bool:
            try:
                await asyncio.wait_for(probe, timeout=5)
                return True
            except Exception:
                return False

        qdrant, postgres = await asyncio.gather(
            check(self.client.get_collections()),
            check(self.doc_store.document_status("readyz-probe")))
        return {"qdrant": qdrant, "postgres": postgres}

    async def delete_document(self, doc_id: str) -> bool:
        """Remove a document's chunks from all three stores. False if it does not exist."""
        if await self.doc_store.document_status(doc_id) in (None, "deleted"):
            return False
        ids = sorted(await self.doc_store.chunk_ids(doc_id))
        # Hide it from retrieval first: get_many only returns chunks of 'ready' documents.
        await self.doc_store.set_status(doc_id, "deleted")
        if ids:
            await self.vector_store.delete(ids)
            await self.sparse_index.delete(ids)
            await self.doc_store.delete_chunks(ids)
        return True

    async def close(self) -> None:
        await self.doc_store.close()
        await self.client.close()


def _split(names: str) -> list[str]:
    return [n.strip() for n in names.split(",") if n.strip()]


async def _build_llms(s: Settings):
    """The generator and critic clients for the configured provider."""
    llm_opts = dict(thinking_level=s.llm_thinking_level,
                    fallback_cooldown_s=s.llm_fallback_cooldown_s)
    if s.llm_provider == "ollama":
        # The two share one client when they name the same model, so the concurrency cap
        # applies across generator and critic calls together.
        llm_opts.update(base_url=s.ollama_url, max_concurrency=s.ollama_max_concurrency,
                        max_retries=s.llm_max_retries, timeout_s=s.ollama_timeout_s,
                        num_thread=s.ollama_num_thread, num_ctx=s.ollama_num_ctx,
                        keep_alive=s.ollama_keep_alive)
        generator = OllamaClient(s.generator_model,
                                 fallback_models=_split(s.generator_fallback_models), **llm_opts)
        critic_llm = generator if (s.critic_model, s.critic_fallback_models) == \
            (s.generator_model, s.generator_fallback_models) else \
            OllamaClient(s.critic_model, fallback_models=_split(s.critic_fallback_models),
                         **llm_opts)
        try:
            for llm in {generator, critic_llm}:
                await llm.check_installed()     # lists local models only; never pulls
        except httpx.TransportError as exc:
            raise RuntimeError(f"Ollama is not reachable at {s.ollama_url}; start it with "
                               f"`ollama serve`") from exc
        return generator, critic_llm

    llm_opts.update(max_concurrency=s.llm_max_concurrency, max_retries=s.llm_max_retries,
                    timeout_s=s.llm_timeout_s)
    if s.llm_provider == "groq":
        llm_cls, api_key = GroqClient, s.groq_api_key
        llm_opts["reasoning_headroom"] = s.llm_reasoning_headroom
    else:
        llm_cls, api_key = GeminiClient, s.gemini_api_key
    if not api_key:
        raise RuntimeError(f"{s.llm_provider.upper()}_API_KEY is not set")
    generator = llm_cls(api_key, s.generator_model,
                        fallback_models=_split(s.generator_fallback_models), **llm_opts)
    critic_llm = llm_cls(api_key, s.critic_model,
                         fallback_models=_split(s.critic_fallback_models), **llm_opts)
    return generator, critic_llm


async def build_services(s: Settings) -> Services:
    """The same wiring as the ingestion CLI, plus retrieval, critics and generation."""
    embedder = SentenceTransformerEmbedder(s.embedding_model)
    client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    vector_store = QdrantVectorStore(client, f"{s.qdrant_collection}_dense")
    sparse_index = QdrantSparseIndex(client, f"{s.qdrant_collection}_sparse", s.sparse_model)
    doc_store = PostgresDocStore(s.database_url)
    await vector_store.setup(embedder.dimension)
    await sparse_index.setup()
    await doc_store.setup()

    retriever = HybridRetriever(embedder, vector_store, sparse_index, doc_store,
                                CrossEncoderReranker(s.reranker_model),
                                dense_top_k=s.dense_top_k, sparse_top_k=s.sparse_top_k,
                                rrf_k=s.rrf_k, fused_top_n=s.fused_top_n)
    generator, critic_llm = await _build_llms(s)
    critic = Critic(critic_llm, s.critic_prob_mode, s.critic_n_samples)
    base = SelfRAGConfig.from_settings(s)
    rags = {mode: SelfRAG(critic, generator, critic_llm, retriever,
                          replace(base, segment_unit=mode)) for mode in MODES}

    pipeline = IngestionPipeline(
        embedder, vector_store, sparse_index, doc_store,
        ChunkingConfig(s.chunk_min_tokens, s.chunk_max_tokens, s.chunk_overlap_ratio))
    return Services(rags, s.segment_unit, pipeline, client, vector_store, sparse_index,
                    doc_store)
