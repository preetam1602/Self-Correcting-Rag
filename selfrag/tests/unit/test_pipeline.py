import asyncio

import pytest

from selfrag.ingestion.pipeline import ChunkingConfig, IngestionPipeline

CHUNKING = ChunkingConfig(min_tokens=5, max_tokens=20, overlap_ratio=0.0)


class FakeEmbedder:
    dimension = 2

    def count_tokens(self, text):
        return len(text.split())

    async def embed_documents(self, texts):
        return [[0.0, 1.0] for _ in texts]


class FakeIndex:
    def __init__(self):
        self.ids, self.fail = set(), False

    async def upsert(self, chunks, vectors=None):
        if self.fail:
            raise RuntimeError("index down")
        self.ids |= {c.chunk_id for c in chunks}

    async def delete(self, chunk_ids):
        self.ids -= set(chunk_ids)


class FakeDocStore:
    def __init__(self):
        self.status, self.chunks = {}, {}

    async def document_status(self, doc_id):
        return self.status.get(doc_id)

    async def chunk_ids(self, doc_id):
        return {cid for cid, c in self.chunks.items() if c.doc_id == doc_id}

    async def begin_ingest(self, doc, chunks):
        self.status[doc.doc_id] = "indexing"
        self.chunks.update({c.chunk_id: c for c in chunks})

    async def delete_chunks(self, chunk_ids):
        for cid in chunk_ids:
            self.chunks.pop(cid, None)

    async def set_status(self, doc_id, status):
        self.status[doc_id] = status


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Leave\n\nAnnual leave is 20 days.\n\n# Pay\n\nPaid monthly.\n",
                    encoding="utf-8")
    dense, sparse, store = FakeIndex(), FakeIndex(), FakeDocStore()
    pipeline = IngestionPipeline(FakeEmbedder(), dense, sparse, store, CHUNKING)

    def ingest():
        return asyncio.run(pipeline.ingest_file(path, tenant_id="t", collection="c", acl=[]))
    return path, ingest, dense, sparse, store


def test_first_ingest_writes_all_three_stores(setup):
    _, ingest, dense, sparse, store = setup
    result = ingest()
    assert (result.status, result.added, result.removed) == ("ready", 2, 0)
    assert dense.ids == sparse.ids == set(store.chunks)
    assert store.status[result.doc_id] == "ready"


def test_unchanged_file_is_skipped(setup):
    _, ingest, *_ = setup
    ingest()
    result = ingest()
    assert (result.status, result.added, result.removed, result.unchanged) == ("unchanged", 0, 0, 2)


def test_edit_reindexes_only_the_changed_chunk(setup):
    path, ingest, dense, sparse, store = setup
    ingest()
    path.write_text("# Leave\n\nAnnual leave is 25 days.\n\n# Pay\n\nPaid monthly.\n",
                    encoding="utf-8")
    result = ingest()
    assert (result.added, result.removed, result.unchanged) == (1, 1, 1)
    assert dense.ids == sparse.ids == set(store.chunks)
    assert any("25 days" in c.text for c in store.chunks.values())
    assert not any("20 days" in c.text for c in store.chunks.values())


def test_failed_run_is_marked_and_repaired_on_retry(setup):
    _, ingest, dense, sparse, store = setup
    sparse.fail = True
    with pytest.raises(RuntimeError):
        ingest()
    assert list(store.status.values()) == ["failed"]

    sparse.fail = False
    result = ingest()           # doc store already has the chunks; indexes must still get them
    assert (result.status, result.added) == ("ready", 2)
    assert dense.ids == sparse.ids == set(store.chunks)
