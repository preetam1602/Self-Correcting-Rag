"""Hybrid retrieval: dense + BM25 per query, fused with RRF, then reranked by a cross-encoder."""
import asyncio
from typing import Sequence

from selfrag.models import Chunk, Filter
from selfrag.providers.interfaces import DocStore, Embedder, Reranker, SparseIndex, VectorStore
from selfrag.retrieval.rrf import rrf


class HybridRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, sparse_index: SparseIndex,
                 doc_store: DocStore, reranker: Reranker, *, dense_top_k: int = 50,
                 sparse_top_k: int = 50, rrf_k: int = 60, fused_top_n: int = 40):
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.doc_store = doc_store
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.fused_top_n = fused_top_n

    async def dense_search(self, query: str, filters: Filter) -> list[str]:
        vector = await self.embedder.embed_query(query)
        return await self.vector_store.search(vector, self.dense_top_k, filters)

    async def sparse_search(self, query: str, filters: Filter) -> list[str]:
        return await self.sparse_index.search(query, self.sparse_top_k, filters)

    async def hybrid_retrieve(self, queries: Sequence[str], filters: Filter) -> list[str]:
        """Dense and BM25 search for every query in parallel, fused into one ranked ID list."""
        searches = [s for q in queries
                    for s in (self.dense_search(q, filters), self.sparse_search(q, filters))]
        ranked_lists = await asyncio.gather(*searches)
        return rrf(ranked_lists, k=self.rrf_k)[:self.fused_top_n]

    async def retrieve(self, query: str, filters: Filter, top_n: int,
                       variants: Sequence[str] = ()) -> list[tuple[Chunk, float]]:
        """Search the query and its paraphrases; rerank against the original query."""
        ids = await self.hybrid_retrieve([query, *variants], filters)
        chunks = await self.doc_store.get_many(ids)
        return await self.reranker.rerank(query, chunks, top_n)
