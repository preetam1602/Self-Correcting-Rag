"""Tenant and ACL filters against a real Qdrant (skipped when Qdrant is not reachable)."""
import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from qdrant_client import AsyncQdrantClient

from selfrag.config import Settings
from selfrag.models import Chunk, Filter
from selfrag.providers.qdrant import QdrantSparseIndex, QdrantVectorStore

# name -> (tenant, collection, acl)
CHUNKS = {
    "public":    ("t1", "hr", []),
    "hr_team":   ("t1", "hr", ["hr-team"]),
    "execs":     ("t1", "hr", ["finance", "execs"]),
    "other_tnt": ("t2", "hr", []),
    "it_public": ("t1", "it", []),
}
IDS = {name: str(uuid.uuid4()) for name in CHUNKS}
NAMES = {v: k for k, v in IDS.items()}


def make_chunks() -> list[Chunk]:
    now = datetime.now(timezone.utc)
    return [Chunk(IDS[name], f"doc_{name}", "Contractor notice period is 30 days.", "T", None, None,
                  "file:///x", tenant, collection, acl, "h", now)
            for name, (tenant, collection, acl) in CHUNKS.items()]


CASES = [
    (Filter(tenant_id="t1"), {"public", "it_public"}),
    (Filter(tenant_id="t1", collection="hr"), {"public"}),
    (Filter(tenant_id="t1", collection="hr", acl_groups=("hr-team",)), {"public", "hr_team"}),
    (Filter(tenant_id="t1", collection="hr", acl_groups=("execs", "sales")), {"public", "execs"}),
    (Filter(tenant_id="t1", acl_groups=("hr-team", "finance")),
     {"public", "hr_team", "execs", "it_public"}),
    (Filter(tenant_id="t2"), {"other_tnt"}),
    (Filter(tenant_id="t3", acl_groups=("hr-team",)), set()),
]


async def run_cases() -> list[tuple[set[str], set[str], set[str]]]:
    s = Settings()
    client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    suffix = uuid.uuid4().hex[:8]
    dense = QdrantVectorStore(client, f"test_f5_dense_{suffix}")
    sparse = QdrantSparseIndex(client, f"test_f5_sparse_{suffix}", s.sparse_model)
    try:
        await dense.setup(dimension=4)
        await sparse.setup()
        chunks = make_chunks()
        await dense.upsert(chunks, [[1.0, 0.0, 0.0, 0.0]] * len(chunks))
        await sparse.upsert(chunks)
        results = []
        for filters, expected in CASES:
            d = {NAMES[i] for i in await dense.search([1.0, 0.0, 0.0, 0.0], 10, filters)}
            b = {NAMES[i] for i in await sparse.search("notice period", 10, filters)}
            results.append((expected, d, b))
        return results
    finally:
        for name in (dense.name, sparse.name):
            if await client.collection_exists(name):
                await client.delete_collection(name)
        await client.close()


def qdrant_reachable() -> bool:
    try:
        s = Settings()
        client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=2)
        asyncio.run(client.get_collections())
        return True
    except Exception:
        return False


@pytest.mark.skipif(not qdrant_reachable(), reason="Qdrant not reachable")
def test_dense_and_sparse_search_respect_tenant_collection_and_acl():
    for (filters, _), (expected, dense, sparse) in zip(CASES, asyncio.run(run_cases())):
        assert dense == expected, filters
        assert sparse == expected, filters
