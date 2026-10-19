"""Qdrant adapters: one collection for dense vectors, one for BM25 sparse vectors.

Points hold only the vector and the filter fields; chunk text lives in the doc store.
"""
import asyncio
from typing import Sequence

from qdrant_client import AsyncQdrantClient, models

from selfrag.models import Chunk, Filter

FILTER_FIELDS = ("doc_id", "tenant_id", "collection", "acl")
BATCH_SIZE = 256


def _payload(c: Chunk) -> dict:
    return {"doc_id": c.doc_id, "tenant_id": c.tenant_id,
            "collection": c.collection, "acl": c.acl}


def query_filter(filters: Filter) -> models.Filter:
    """Tenant (+ collection) must match; ACL must be empty or share a group with the caller."""
    must: list = [models.FieldCondition(key="tenant_id",
                                        match=models.MatchValue(value=filters.tenant_id))]
    if filters.collection is not None:
        must.append(models.FieldCondition(key="collection",
                                          match=models.MatchValue(value=filters.collection)))
    acl_ok: list = [models.IsEmptyCondition(is_empty=models.PayloadField(key="acl"))]
    if filters.acl_groups:
        acl_ok.append(models.FieldCondition(key="acl",
                                            match=models.MatchAny(any=list(filters.acl_groups))))
    must.append(models.Filter(should=acl_ok))
    return models.Filter(must=must)


async def _create_payload_indexes(client: AsyncQdrantClient, name: str) -> None:
    for field in FILTER_FIELDS:
        await client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


async def _upsert_batched(client: AsyncQdrantClient, name: str,
                          points: list[models.PointStruct]) -> None:
    for i in range(0, len(points), BATCH_SIZE):
        await client.upsert(name, points=points[i:i + BATCH_SIZE], wait=True)


async def _delete(client: AsyncQdrantClient, name: str, chunk_ids: Sequence[str]) -> None:
    if chunk_ids:
        await client.delete(name, points_selector=models.PointIdsList(points=list(chunk_ids)),
                            wait=True)


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient, name: str):
        self.client, self.name = client, name

    async def setup(self, dimension: int) -> None:
        if not await self.client.collection_exists(self.name):
            await self.client.create_collection(
                self.name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE))
            await _create_payload_indexes(self.client, self.name)

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        points = [models.PointStruct(id=c.chunk_id, vector=v, payload=_payload(c))
                  for c, v in zip(chunks, vectors)]
        await _upsert_batched(self.client, self.name, points)

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        await _delete(self.client, self.name, chunk_ids)

    async def search(self, vector: list[float], k: int, filters: Filter) -> list[str]:
        result = await self.client.query_points(
            self.name, query=vector, limit=k, query_filter=query_filter(filters),
            with_payload=False)
        return [str(p.id) for p in result.points]


class QdrantSparseIndex:
    """BM25 as Qdrant sparse vectors: term frequencies from fastembed, IDF applied by Qdrant."""

    VECTOR = "bm25"

    def __init__(self, client: AsyncQdrantClient, name: str, model_name: str):
        from fastembed import SparseTextEmbedding

        self.client, self.name = client, name
        self._bm25 = SparseTextEmbedding(model_name)

    async def setup(self) -> None:
        if not await self.client.collection_exists(self.name):
            await self.client.create_collection(
                self.name, vectors_config={},
                sparse_vectors_config={
                    self.VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)})
            await _create_payload_indexes(self.client, self.name)

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        embeddings = await asyncio.to_thread(
            lambda: list(self._bm25.embed([c.text for c in chunks])))
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector={self.VECTOR: models.SparseVector(indices=e.indices.tolist(),
                                                         values=e.values.tolist())},
                payload=_payload(c))
            for c, e in zip(chunks, embeddings)]
        await _upsert_batched(self.client, self.name, points)

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        await _delete(self.client, self.name, chunk_ids)

    async def search(self, query: str, k: int, filters: Filter) -> list[str]:
        e = await asyncio.to_thread(lambda: next(iter(self._bm25.query_embed(query))))
        result = await self.client.query_points(
            self.name,
            query=models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist()),
            using=self.VECTOR, limit=k, query_filter=query_filter(filters), with_payload=False)
        return [str(p.id) for p in result.points]
