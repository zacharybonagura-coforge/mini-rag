import json
import math
from pathlib import Path

from embeddings.base import Chunk, EmbeddingAdapter, ScoredChunk
from embeddings.ollama import OllamaEmbeddingAdapter


def load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text())
    return [Chunk.model_validate(item) for item in data]


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute cosine distance for a zero vector")
    cosine_similarity = dot / (norm_a * norm_b)
    return 1.0 - cosine_similarity


def retrieve(
    query: str,
    chunks: list[Chunk],
    adapter: EmbeddingAdapter,
    k: int = 3,
) -> list[ScoredChunk]:
    query_vector = adapter.embed_query(query)
    scored = [
        ScoredChunk(
            chunk=chunk,
            score=cosine_distance(query_vector, chunk.embedding),
        )
        for chunk in chunks
    ]
    scored.sort(key=lambda item: item.score)
    return scored[:k]


if __name__ == "__main__":
    adapter = OllamaEmbeddingAdapter()
    chunks = load_chunks(Path("store/chunks.json"))
    question = "How much can I spend on meals?"
    for item in retrieve(question, chunks, adapter):
        print(f"{item.score:.4f}  {item.chunk.chunk_id}  {item.chunk.section_title}")
        print(item.chunk.text)
        print()