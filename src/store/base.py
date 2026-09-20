from typing import Protocol

from embeddings.base import Chunk, ScoredChunk


class VectorStoreAdapter(Protocol):
    provider: str

    def save_chunks(self, chunks: list[Chunk]) -> None: ...

    def load_chunks(self) -> list[Chunk]: ...

    def search(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[ScoredChunk]: ...
