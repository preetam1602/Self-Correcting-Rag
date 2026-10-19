import asyncio
from typing import Sequence

from selfrag.models import Chunk


class CrossEncoderReranker:
    """Local cross-encoder reranker (default BAAI/bge-reranker-v2-m3). Scores are in 0..1."""

    def __init__(self, model_name: str, max_length: int = 1024, batch_size: int = 32):
        import torch
        from sentence_transformers import CrossEncoder
        from transformers import AutoTokenizer

        # Small rerankers (e.g. MiniLM) only take 512 tokens; longer input crashes them.
        max_length = min(max_length, AutoTokenizer.from_pretrained(model_name).model_max_length)
        self._model = CrossEncoder(model_name, max_length=max_length,
                                   activation_fn=torch.nn.Sigmoid())
        self._batch_size = batch_size

    async def rerank(self, query: str, chunks: Sequence[Chunk],
                     top_n: int) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        scores = await asyncio.to_thread(
            self._model.predict, [(query, c.text) for c in chunks],
            batch_size=self._batch_size, show_progress_bar=False)
        ranked = sorted(zip(chunks, scores.tolist()), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
