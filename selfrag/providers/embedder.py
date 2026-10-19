import asyncio
from typing import Sequence


class SentenceTransformerEmbedder:
    """Local dense embedder (default BAAI/bge-m3). Its tokenizer also sizes chunks."""

    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(
            self._model.encode, list(texts),
            batch_size=self._batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
