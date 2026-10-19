"""PostgreSQL doc store: source of truth for chunk text, metadata and document status."""
from datetime import datetime, timezone
from typing import Sequence

import asyncpg

from selfrag.models import Chunk, Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    collection  TEXT NOT NULL,
    title       TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    acl         TEXT[] NOT NULL DEFAULT '{}',
    version     INTEGER NOT NULL DEFAULT 1,
    status      TEXT NOT NULL,              -- indexing | ready | failed
    updated_at  TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    text         TEXT NOT NULL,
    title        TEXT NOT NULL,
    section      TEXT,
    page         INTEGER,
    source_url   TEXT NOT NULL,
    tenant_id    TEXT NOT NULL,
    collection   TEXT NOT NULL,
    acl          TEXT[] NOT NULL DEFAULT '{}',
    updated_at   TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
"""


class PostgresDocStore:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def setup(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def get_many(self, ids: Sequence[str]) -> list[Chunk]:
        """Chunks in the order of `ids`, skipping documents that are not 'ready'."""
        rows = await self._pool.fetch(
            """
            SELECT c.chunk_id, c.doc_id, c.text, c.title, c.section, c.page, c.source_url,
                   c.tenant_id, c.collection, c.acl, c.content_hash, c.updated_at
            FROM chunks c JOIN documents d USING (doc_id)
            WHERE c.chunk_id = ANY($1::text[]) AND d.status = 'ready'
            """,
            list(ids))
        by_id = {r["chunk_id"]: Chunk(**dict(r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    async def document_status(self, doc_id: str) -> str | None:
        return await self._pool.fetchval("SELECT status FROM documents WHERE doc_id = $1", doc_id)

    async def chunk_ids(self, doc_id: str) -> set[str]:
        rows = await self._pool.fetch("SELECT chunk_id FROM chunks WHERE doc_id = $1", doc_id)
        return {r["chunk_id"] for r in rows}

    async def begin_ingest(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        """Mark the document as indexing (new version) and write its new chunks."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO documents (doc_id, tenant_id, collection, title, source_url, acl,
                                       status, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, 'indexing', $7)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title, source_url = EXCLUDED.source_url,
                    acl = EXCLUDED.acl, status = 'indexing',
                    version = documents.version + 1, updated_at = EXCLUDED.updated_at
                """,
                doc.doc_id, doc.tenant_id, doc.collection, doc.title, doc.source_url, doc.acl,
                datetime.now(timezone.utc))
            await conn.executemany(
                """
                INSERT INTO chunks (chunk_id, doc_id, content_hash, text, title, section, page,
                                    source_url, tenant_id, collection, acl, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                [(c.chunk_id, c.doc_id, c.content_hash, c.text, c.title, c.section, c.page,
                  c.source_url, c.tenant_id, c.collection, c.acl, c.updated_at) for c in chunks])

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        if chunk_ids:
            await self._pool.execute("DELETE FROM chunks WHERE chunk_id = ANY($1::text[])",
                                     list(chunk_ids))

    async def set_status(self, doc_id: str, status: str) -> None:
        await self._pool.execute(
            "UPDATE documents SET status = $2, updated_at = $3 WHERE doc_id = $1",
            doc_id, status, datetime.now(timezone.utc))
