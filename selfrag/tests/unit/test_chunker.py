from selfrag.ingestion.chunker import chunk_blocks
from selfrag.ingestion.parsers import Block


def words(text: str) -> int:
    return len(text.split())


def sentences(n: int, length: int = 10, start: int = 0) -> str:
    return " ".join(" ".join([f"w{i}"] * (length - 1)) + " end." for i in range(start, start + n))


def test_small_section_is_one_chunk():
    pieces = chunk_blocks([Block("One. Two.", "A", 3)], words)
    assert [(p.text, p.section, p.page) for p in pieces] == [("One. Two.", "A", 3)]


def test_sizes_stay_within_limits_and_overlap():
    # 200 sentences x 10 words = 2000 tokens
    pieces = chunk_blocks([Block(sentences(200), "A", None)], words,
                          min_tokens=400, max_tokens=800, overlap_ratio=0.12)
    sizes = [words(p.text) for p in pieces]
    assert len(pieces) == 3
    assert all(400 <= s <= 800 for s in sizes)
    for prev, nxt in zip(pieces, pieces[1:]):           # overlap: next starts inside prev's tail
        overlap = prev.text[prev.text.index(nxt.text.split(".")[0]):]
        assert nxt.text.startswith(overlap)
        assert 0.05 * words(prev.text) <= words(overlap) <= 0.15 * words(prev.text)


def test_chunks_never_cross_sections():
    blocks = [Block(sentences(30), "A", None), Block(sentences(30, start=100), "B", None)]
    pieces = chunk_blocks(blocks, words)
    assert [p.section for p in pieces] == ["A", "B"]


def test_paragraph_breaks_are_kept():
    pieces = chunk_blocks([Block("First para.", "A", None), Block("Second para.", "A", None)], words)
    assert pieces[0].text == "First para.\n\nSecond para."


def test_oversized_sentence_is_split():
    pieces = chunk_blocks([Block(" ".join(["x"] * 2000), None, None)], words,
                          min_tokens=400, max_tokens=800)
    assert all(words(p.text) <= 800 for p in pieces)
    assert sum(words(p.text) for p in pieces) >= 2000


def test_page_is_first_page_of_chunk():
    blocks = [Block("Page one.", None, 1), Block("Page two.", None, 2)]
    assert chunk_blocks(blocks, words)[0].page == 1
