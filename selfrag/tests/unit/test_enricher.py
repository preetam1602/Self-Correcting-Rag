from dataclasses import replace
from pathlib import Path

from selfrag.ingestion.chunker import ChunkPiece
from selfrag.ingestion.enricher import enrich, make_doc_id
from selfrag.models import Document

DOC = Document("d1", "t1", "hr", "Policy", "file:///policy.md", ["staff"])


def test_ids_are_stable_across_runs():
    pieces = [ChunkPiece("Notice is 30 days.", "Leave", None)]
    assert enrich(DOC, pieces)[0].chunk_id == enrich(DOC, pieces)[0].chunk_id


def test_any_stored_change_changes_the_id():
    piece = ChunkPiece("Notice is 30 days.", "Leave", None)
    base = enrich(DOC, [piece])[0].chunk_id
    assert enrich(DOC, [replace(piece, text="Notice is 60 days.")])[0].chunk_id != base
    assert enrich(DOC, [replace(piece, section="Pay")])[0].chunk_id != base
    assert enrich(replace(DOC, acl=["admins"]), [piece])[0].chunk_id != base
    assert enrich(replace(DOC, title="New title"), [piece])[0].chunk_id != base


def test_duplicate_pieces_are_stored_once():
    piece = ChunkPiece("Same text.", "A", None)
    assert len(enrich(DOC, [piece, piece])) == 1


def test_doc_id_depends_on_tenant_collection_and_path():
    path = Path("/docs/policy.md")
    assert make_doc_id("t1", "hr", path) == make_doc_id("t1", "hr", path)
    assert make_doc_id("t1", "hr", path) != make_doc_id("t2", "hr", path)
    assert make_doc_id("t1", "hr", path) != make_doc_id("t1", "it", path)
