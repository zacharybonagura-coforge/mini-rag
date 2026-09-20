from pathlib import Path

from embeddings.base import Chunk, EmbeddingAdapter
from parse_policy import parse_policy
from store.base import VectorStoreAdapter


def ingest(
    path: Path,
    embedder: EmbeddingAdapter,
    store: VectorStoreAdapter,
) -> list[Chunk]:
    sections = parse_policy(path)
    vectors = embedder.embed_documents([section.text for section in sections])
    chunks = [
        Chunk(**section.model_dump(), embedding=vector)
        for section, vector in zip(sections, vectors, strict=True)
    ]
    store.save_chunks(chunks)
    return chunks


if __name__ == "__main__":
    from adapters import build_embedder, build_store
    from config import load_settings

    settings = load_settings()
    embedder = build_embedder(settings)
    store = build_store(settings)
    chunks = ingest(Path(settings.policy_path), embedder, store)
    print(f"Wrote {len(chunks)} chunks via {store.provider}")
