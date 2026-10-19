"""Turn chunk pieces into Chunk records with metadata, content hash and stable IDs."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from selfrag.ingestion.chunker import ChunkPiece
from selfrag.models import Chunk, Document

_NAMESPACE = uuid.UUID("6f1c3a52-6a3e-4c55-9a43-2b1e5b7f0d11")


def make_doc_id(tenant_id: str, collection: str, path: Path) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{tenant_id}/{collection}/{path.as_posix()}"))


def content_hash(doc: Document, piece: ChunkPiece) -> str:
    """Hash of everything stored with a chunk, so any change re-indexes it."""
    data = [piece.text, piece.section, piece.page, doc.title, doc.source_url,
            doc.tenant_id, doc.collection, sorted(doc.acl)]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def enrich(doc: Document, pieces: list[ChunkPiece]) -> list[Chunk]:
    now = datetime.now(timezone.utc)
    chunks: dict[str, Chunk] = {}
    for piece in pieces:
        digest = content_hash(doc, piece)
        chunk_id = str(uuid.uuid5(_NAMESPACE, f"{doc.doc_id}/{digest}"))
        if chunk_id in chunks:                  # identical chunk twice in one document
            continue
        chunks[chunk_id] = Chunk(
            chunk_id=chunk_id, doc_id=doc.doc_id, text=piece.text, title=doc.title,
            section=piece.section, page=piece.page, source_url=doc.source_url,
            tenant_id=doc.tenant_id, collection=doc.collection, acl=list(doc.acl),
            content_hash=digest, updated_at=now)
    return list(chunks.values())
