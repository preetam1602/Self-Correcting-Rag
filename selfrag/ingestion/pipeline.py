"""Ingest one file: parse -> chunk -> enrich -> diff against the doc store -> write.

Write order: doc store first, then vector and sparse indexes. The document is
marked 'ready' only after all three succeed.
"""
from dataclasses import dataclass
from pathlib import Path

from selfrag.ingestion.chunker import chunk_blocks
from selfrag.ingestion.enricher import enrich, make_doc_id
from selfrag.ingestion.parsers import parse_file
from selfrag.models import Document
from selfrag.providers.interfaces import DocStore, Embedder, SparseIndex, VectorStore


@dataclass(frozen=True)
class ChunkingConfig:
    min_tokens: int = 400
    max_tokens: int = 800
    overlap_ratio: float = 0.12


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    path: str
    status: str             # "ready" | "unchanged"
    added: int
    removed: int
    unchanged: int


class IngestionPipeline:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, sparse_index: SparseIndex,
                 doc_store: DocStore, chunking: ChunkingConfig = ChunkingConfig()):
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.doc_store = doc_store
        self.chunking = chunking

    async def ingest_file(self, path: Path, *, tenant_id: str, collection: str,
                          acl: list[str]) -> IngestResult:
        path = Path(path).resolve()
        parsed = parse_file(path)
        doc = Document(doc_id=make_doc_id(tenant_id, collection, path), tenant_id=tenant_id,
                       collection=collection, title=parsed.title, source_url=path.as_uri(),
                       acl=acl)
        pieces = chunk_blocks(parsed.blocks, self.embedder.count_tokens,
                              min_tokens=self.chunking.min_tokens,
                              max_tokens=self.chunking.max_tokens,
                              overlap_ratio=self.chunking.overlap_ratio)
        chunks = enrich(doc, pieces)

        status = await self.doc_store.document_status(doc.doc_id)
        existing = await self.doc_store.chunk_ids(doc.doc_id)
        new_ids = {c.chunk_id for c in chunks}
        to_remove = sorted(existing - new_ids)
        # After a failed or interrupted run the indexes may lack chunks the doc store
        # already has, so re-write all of them (every write below is idempotent).
        to_add = chunks if status != "ready" else [c for c in chunks if c.chunk_id not in existing]
        unchanged = len(chunks) - len(to_add)

        if status == "ready" and not to_add and not to_remove:
            return IngestResult(doc.doc_id, str(path), "unchanged", 0, 0, unchanged)

        try:
            await self.doc_store.begin_ingest(doc, to_add)
            if to_add:
                vectors = await self.embedder.embed_documents([c.text for c in to_add])
                await self.vector_store.upsert(to_add, vectors)
                await self.sparse_index.upsert(to_add)
            if to_remove:
                await self.vector_store.delete(to_remove)
                await self.sparse_index.delete(to_remove)
                await self.doc_store.delete_chunks(to_remove)
            await self.doc_store.set_status(doc.doc_id, "ready")
        except Exception:
            await self.doc_store.set_status(doc.doc_id, "failed")
            raise
        return IngestResult(doc.doc_id, str(path), "ready", len(to_add), len(to_remove), unchanged)
