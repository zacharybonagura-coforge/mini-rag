import json
from pathlib import Path

from embeddings.base import Chunk, EmbeddingAdapter
from embeddings.ollama import OllamaEmbeddingAdapter
from parse_policy import parse_policy


def ingest(
    path: Path,
    adapter: EmbeddingAdapter,
    out_path: Path,
) -> list[Chunk]:
    sections = parse_policy(path)
    vectors = adapter.embed_documents([section.text for section in sections])
    chunks = []
    for section, vector in zip(sections, vectors, strict=True):
        chunks.append(Chunk(**section.model_dump(), embedding=vector))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))
    return chunks


if __name__ == "__main__":
    chunks = ingest(
        Path("policy.md"),
        OllamaEmbeddingAdapter(),
        Path("store/chunks.json"),
    )
    print(f"Wrote {len(chunks)} chunks to store/chunks.json")