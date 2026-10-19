"""Ingest files or folders from the command line.

    python -m selfrag.ingestion docs/ policy.pdf --collection hr-policies
"""
import argparse
import asyncio
import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from selfrag.config import Settings
from selfrag.ingestion.parsers import PARSERS
from selfrag.ingestion.pipeline import ChunkingConfig, IngestionPipeline
from selfrag.providers.embedder import SentenceTransformerEmbedder
from selfrag.providers.postgres import PostgresDocStore
from selfrag.providers.qdrant import QdrantSparseIndex, QdrantVectorStore


def collect_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix.lower() in PARSERS)
        else:
            files.append(path)
    return files


async def run(args: argparse.Namespace) -> int:
    files = collect_files(args.paths)
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    settings = Settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    vector_store = QdrantVectorStore(client, f"{settings.qdrant_collection}_dense")
    sparse_index = QdrantSparseIndex(client, f"{settings.qdrant_collection}_sparse",
                                     settings.sparse_model)
    doc_store = PostgresDocStore(settings.database_url)
    await vector_store.setup(embedder.dimension)
    await sparse_index.setup()
    await doc_store.setup()

    pipeline = IngestionPipeline(
        embedder, vector_store, sparse_index, doc_store,
        ChunkingConfig(settings.chunk_min_tokens, settings.chunk_max_tokens,
                       settings.chunk_overlap_ratio))
    failures = 0
    try:
        for path in files:
            try:
                r = await pipeline.ingest_file(path, tenant_id=args.tenant,
                                               collection=args.collection, acl=args.acl)
                print(f"{r.status:9} {path}  (+{r.added} -{r.removed} ={r.unchanged})")
            except Exception as exc:
                failures += 1
                print(f"failed    {path}: {exc}", file=sys.stderr)
    finally:
        await doc_store.close()
        await client.close()
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m selfrag.ingestion",
                                     description="Ingest PDF, HTML, DOCX and Markdown files.")
    parser.add_argument("paths", nargs="+", type=Path, help="files or folders")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--collection", default="default")
    parser.add_argument("--acl", default=[], type=lambda s: [a for a in s.split(",") if a],
                        help="comma-separated ACL groups")
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
