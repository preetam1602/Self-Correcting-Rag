import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from selfrag.models import Chunk, Filter
from selfrag.retrieval.hybrid_retrieve import HybridRetriever
from selfrag.retrieval.rrf import rrf


def test_rrf_scores_by_reciprocal_rank():
    # a: 1/61 + 1/62, b: 1/62 + 1/61, c: 1/63, d: 1/61
    assert rrf([["a", "b", "c"], ["b", "a"], ["d"]], k=60) == ["a", "b", "d", "c"]


def test_rrf_rewards_agreement_between_lists():
    # "x" is 3rd in both lists; "y" is 1st in one list only
    assert rrf([["y", "p", "x"], ["q", "r", "x"]], k=60)[0] == "x"


def test_rrf_empty():
    assert rrf([[], []]) == []


CHUNK = Chunk("", "d1", "", "T", None, None, "file:///t", "t1", "hr", [], "h",
              datetime(2026, 1, 1, tzinfo=timezone.utc))


class FakeEmbedder:
    async def embed_query(self, text):
        return [float(len(text))]


class FakeSearch:
    def __init__(self, results):
        self.results, self.calls = results, []

    async def search(self, query_or_vector, k, filters):
        self.calls.append((query_or_vector, k, filters))
        return self.results[:k]


class FakeDocStore:
    def __init__(self, known):
        self.known = known

    async def get_many(self, ids):
        return [replace(CHUNK, chunk_id=i, text=f"text {i}") for i in ids if i in self.known]


class FakeReranker:
    async def rerank(self, query, chunks, top_n):
        ranked = sorted(chunks, key=lambda c: c.chunk_id)      # deterministic "relevance"
        return [(c, 0.5) for c in ranked][:top_n]


@pytest.fixture
def retriever():
    dense = FakeSearch(["a", "b", "c"])
    sparse = FakeSearch(["c", "d"])
    r = HybridRetriever(FakeEmbedder(), dense, sparse, FakeDocStore({"a", "b", "c"}),
                        FakeReranker(), dense_top_k=3, sparse_top_k=2, rrf_k=60, fused_top_n=3)
    return r, dense, sparse


def test_hybrid_fuses_dense_and_sparse_and_cuts_to_top_n(retriever):
    r, dense, sparse = retriever
    ids = asyncio.run(r.hybrid_retrieve(["q"], Filter(tenant_id="t", collection="hr")))
    assert ids == ["c", "a", "b"]                       # c is in both lists
    assert dense.calls == [([1.0], 3, Filter(tenant_id="t", collection="hr"))]
    assert sparse.calls == [("q", 2, Filter(tenant_id="t", collection="hr"))]


def test_hybrid_searches_every_query(retriever):
    r, dense, sparse = retriever
    asyncio.run(r.hybrid_retrieve(["q1", "q22"], Filter(tenant_id="t")))
    assert [c[0] for c in dense.calls] == [[2.0], [3.0]]
    assert [c[0] for c in sparse.calls] == ["q1", "q22"]


def test_retrieve_fetches_text_and_reranks(retriever):
    r, *_ = retriever
    ranked = asyncio.run(r.retrieve("q", Filter(tenant_id="t"), top_n=2))
    assert [(c.chunk_id, c.text) for c, _ in ranked] == [("a", "text a"), ("b", "text b")]


def test_filter_requires_a_tenant():
    with pytest.raises(TypeError):
        Filter()
    with pytest.raises(ValueError):
        Filter(tenant_id="")
