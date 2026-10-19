"""Render the best beam as answer text with [n] markers plus one citation per segment."""
import re

from selfrag.models import AnswerSegment, Beam, Citation

_TRAILING_PUNCT = re.compile(r"^(.*?)([.!?]+[\"')\]]*)$", re.DOTALL)


def with_marker(text: str, n: int) -> str:
    """'Notice is 30 days.' -> 'Notice is 30 days [1].'"""
    match = _TRAILING_PUNCT.match(text)
    return f"{match.group(1)} [{n}]{match.group(2)}" if match else f"{text} [{n}]"


def render_with_citations(beam: Beam) -> tuple[str, list[AnswerSegment]]:
    numbers: dict[str, int] = {}            # chunk_id -> n, numbered by first use
    parts, segments = [], []
    for text, chunk, support in zip(beam.segments, beam.citations, beam.supports):
        if chunk is None:                   # written without evidence: no citation
            parts.append(text)
            segments.append(AnswerSegment(text=text, citation=None))
            continue
        n = numbers.setdefault(chunk.chunk_id, len(numbers) + 1)
        parts.append(with_marker(text, n))
        segments.append(AnswerSegment(text=text, citation=Citation(
            n=n, doc_id=chunk.doc_id, title=chunk.title, page=chunk.page,
            url=chunk.source_url, support=support)))
    return " ".join(parts), segments
