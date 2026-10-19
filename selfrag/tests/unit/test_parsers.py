import pytest

from selfrag.ingestion.parsers import Block, parse_file
from selfrag.tests.fixtures import write_docx, write_pdf


def test_markdown_sections_and_code_fence(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text(
        "# Handbook\n\nIntro text.\n\n## Leave\n\nAnnual leave is 20 days.\n"
        "Second line of the same paragraph.\n\n```\n# not a heading\n\nstill code\n```\n\n"
        "### Sick\n\nSick leave is 10 days.\n\n# Pay\n\nMonthly.\n", encoding="utf-8")
    doc = parse_file(path)
    assert doc.title == "Handbook"
    assert doc.blocks == [
        Block("Intro text.", "Handbook", None),
        Block("Annual leave is 20 days.\nSecond line of the same paragraph.", "Handbook > Leave", None),
        Block("```\n# not a heading\n\nstill code\n```", "Handbook > Leave", None),
        Block("Sick leave is 10 days.", "Handbook > Leave > Sick", None),
        Block("Monthly.", "Pay", None),
    ]


def test_markdown_without_h1_uses_file_name(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("Just text.\n", encoding="utf-8")
    assert parse_file(path).title == "notes"


def test_html_blocks_tables_and_dropped_tags(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Policy</title><style>p{}</style></head><body>"
        "<nav><a>Home</a></nav><h1>Contractors</h1><div><p>Notice is 30 days.</p></div>"
        "<h2>Rates</h2><table><tr><th>Type</th><th>Days</th></tr>"
        "<tr><td>Invoice</td><td>15</td></tr></table>"
        "<ul><li>First item</li></ul><script>alert(1)</script></body></html>", encoding="utf-8")
    doc = parse_file(path)
    assert doc.title == "Policy"
    assert doc.blocks == [
        Block("Notice is 30 days.", "Contractors", None),
        Block("Type | Days", "Contractors > Rates", None),
        Block("Invoice | 15", "Contractors > Rates", None),
        Block("First item", "Contractors > Rates", None),
    ]


def test_docx_title_headings_and_table(tmp_path):
    doc = parse_file(write_docx(tmp_path / "handbook.docx"))
    assert doc.title == "Contractor Handbook"
    assert doc.blocks == [
        Block("Contractors must give 30 days notice.", "Notice", None),
        Block("Type | Days", "Notice > Payment", None),
        Block("Invoice | 15", "Notice > Payment", None),
    ]


def test_pdf_pages(tmp_path):
    doc = parse_file(write_pdf(tmp_path / "policy.pdf", ["Page one text.", "Page two text."]))
    assert doc.title == "policy"
    assert doc.blocks == [Block("Page one text.", None, 1), Block("Page two text.", None, 2)]


def test_unsupported_type(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported file type"):
        parse_file(path)


# ---------------------------------------------------------------- PDF text cleanup

from selfrag.ingestion.parsers import _repeated_lines, clean_pdf_text, pdf_vocabulary


def test_pdf_cleanup_fixes_small_caps_hyphens_and_ligatures():
    pages = ["6 C ONCLUSION\nSELF -R AG is new. Self-Reﬂective Retrieval-Augmented Gen-\n"
             "eration (SELF -\nRAG) uses retrieval-\naugmented passages. A new conclusion.\n"
             "Retrieval-\nAugmented text. B EXPERIMENTAL details are experimental. RAG."]
    vocab = pdf_vocabulary(pages + ["retrieval-augmented generation"])
    assert clean_pdf_text(pages[0], set(), vocab) == (
        "6 CONCLUSION\nSELF-RAG is new. Self-Reflective Retrieval-Augmented Generation "
        "(SELF-RAG) uses retrieval-augmented passages. A new conclusion.\n"
        "Retrieval-Augmented text. B EXPERIMENTAL details are experimental. RAG.")


def test_pdf_cleanup_drops_running_headers_and_numeric_lines():
    pages = [f"Preprint.\nBody of page {i}.\n0 50 100 150\n{i}" for i in range(1, 5)]
    drop = _repeated_lines(pages)
    assert "Preprint." in drop
    assert clean_pdf_text(pages[1], drop, pdf_vocabulary(pages)) == "Body of page 2."


def test_short_pdfs_keep_every_line():
    assert _repeated_lines(["Title\nA", "Title\nB"]) == set()
