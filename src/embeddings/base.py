from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    text: str
    embedding: list[float]

class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float

class EmbeddingAdapter(Protocol):
    provider: str
    model_id: str
    dimension: int

    def embed_query(
        self,
        query: str
    ) -> list[float]: ...

    def embed_documents(
        self,
        texts: Sequence[str]
    ) -> list[list[float]]: ...
