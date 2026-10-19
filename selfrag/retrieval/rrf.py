from collections import defaultdict
from typing import Sequence


def rrf(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: score(id) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)
