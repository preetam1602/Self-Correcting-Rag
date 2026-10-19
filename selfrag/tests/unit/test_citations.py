from dataclasses import replace
from datetime import datetime, timezone

from selfrag.generation.citations import render_with_citations, with_marker
from selfrag.models import Beam, Chunk

BASE = Chunk("c1", "d_812", "", "Contractor Policy", None, 4, "https://kb/policy", "t", "hr", [],
             "h", datetime(2026, 1, 1, tzinfo=timezone.utc))
OTHER = replace(BASE, chunk_id="c2", doc_id="d_900", title="Handbook", page=None,
                source_url="file:///handbook.docx")


def test_marker_goes_before_final_punctuation():
    assert with_marker("Notice is 30 days.", 1) == "Notice is 30 days [1]."
    assert with_marker('He said "yes."', 2) == 'He said "yes [2]."'
    assert with_marker("Is it?!", 3) == "Is it [3]?!"
    assert with_marker("No punctuation", 4) == "No punctuation [4]"


def test_render_numbers_passages_by_first_use_and_reuses_numbers():
    beam = Beam(["Notice is 30 days.", "Pay is monthly.", "Rates are fixed.", "Ask HR."],
                [BASE, OTHER, BASE, None], [1.0, 1.0, 1.0, 1.0],
                ["fully", "partially", "fully", None])
    answer, segments = render_with_citations(beam)
    assert answer == "Notice is 30 days [1]. Pay is monthly [2]. Rates are fixed [1]. Ask HR."
    assert [s.citation.n if s.citation else None for s in segments] == [1, 2, 1, None]
    assert segments[0].text == "Notice is 30 days."
    assert segments[0].citation.model_dump() == {
        "n": 1, "doc_id": "d_812", "title": "Contractor Policy", "page": 4,
        "url": "https://kb/policy", "support": "fully"}
    assert segments[1].citation.support == "partially"
