"""Split parsed blocks into 400-800 token chunks with ~12% overlap.

Chunks never cross a section boundary, so an edit in one section only changes
that section's chunks (which keeps incremental re-indexing small).
"""
import math
import re
from dataclasses import dataclass
from itertools import groupby
from typing import Callable

from selfrag.ingestion.parsers import Block

CountTokens = Callable[[str], int]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkPiece:
    text: str
    section: str | None
    page: int | None


@dataclass(frozen=True)
class _Unit:
    text: str
    block: int          # index of the source block, to keep paragraph breaks
    page: int | None
    tokens: int


def _split_long(text: str, count_tokens: CountTokens, max_tokens: int) -> list[str]:
    """Split a sentence longer than max_tokens into word windows that fit."""
    words, parts = text.split(), []
    while words:
        n = max(1, len(words) * max_tokens // max(count_tokens(" ".join(words)), 1))
        while n > 1 and count_tokens(" ".join(words[:n])) > max_tokens:
            n = max(1, int(n * 0.9))
        parts.append(" ".join(words[:n]))
        words = words[n:]
    return parts


def _units(blocks: list[Block], first_index: int, count_tokens: CountTokens,
           max_tokens: int) -> list[_Unit]:
    units = []
    for i, block in enumerate(blocks, start=first_index):
        text = " ".join(block.text.split())
        for sentence in _SENTENCE_END.split(text):
            if not sentence:
                continue
            tokens = count_tokens(sentence)
            parts = [sentence] if tokens <= max_tokens else _split_long(sentence, count_tokens, max_tokens)
            for part in parts:
                units.append(_Unit(part, i, block.page,
                                   tokens if len(parts) == 1 else count_tokens(part)))
    return units


def _render(units: list[_Unit], section: str | None) -> ChunkPiece:
    text = units[0].text
    for prev, unit in zip(units, units[1:]):
        text += (" " if unit.block == prev.block else "\n\n") + unit.text
    return ChunkPiece(text, section, units[0].page)


def _overlap_tail(units: list[_Unit], ratio: float) -> list[_Unit]:
    budget = ratio * sum(u.tokens for u in units)
    tail, used = [], 0
    for unit in reversed(units[1:]):            # never carry the whole chunk over
        if used + unit.tokens > budget:
            break
        tail.insert(0, unit)
        used += unit.tokens
    return tail


def _pack(units: list[_Unit], section: str | None, min_tokens: int, max_tokens: int,
          overlap_ratio: float) -> list[ChunkPiece]:
    # Aim for evenly sized chunks (overlap included) so the last one is not a tiny remainder.
    total = sum(u.tokens for u in units)
    n = max(1, math.ceil(total / (max_tokens * (1 - overlap_ratio))))
    target = min(max_tokens,
                 max(min_tokens, math.ceil(total / (1 + (n - 1) * (1 - overlap_ratio)))))
    chunks, current, size = [], [], 0
    for unit in units:
        if current and (size >= target or size + unit.tokens > max_tokens):
            chunks.append(_render(current, section))
            current = _overlap_tail(current, overlap_ratio)
            size = sum(u.tokens for u in current)
            if size + unit.tokens > max_tokens:
                current, size = [], 0
        current.append(unit)
        size += unit.tokens
    if current:
        chunks.append(_render(current, section))
    return chunks


def chunk_blocks(blocks: list[Block], count_tokens: CountTokens, *, min_tokens: int = 400,
                 max_tokens: int = 800, overlap_ratio: float = 0.12) -> list[ChunkPiece]:
    pieces, index = [], 0
    for section, group in groupby(blocks, key=lambda b: b.section):
        group = list(group)
        units = _units(group, index, count_tokens, max_tokens)
        index += len(group)
        if units:
            pieces.extend(_pack(units, section, min_tokens, max_tokens, overlap_ratio))
    return pieces
