from embeddings.base import EmbeddingAdapter, ScoredChunk
from store.base import VectorStoreAdapter


def retrieve(
    query: str,
    embedder: EmbeddingAdapter,
    store: VectorStoreAdapter,
    k: int = 3,
) -> list[ScoredChunk]:
    query_vector = embedder.embed_query(query)
    return store.search(query_vector, k=k)


if __name__ == "__main__":
    from adapters import build_embedder, build_store
    from config import load_settings

    settings = load_settings()
    embedder = build_embedder(settings)
    store = build_store(settings)
    question = "How much can I spend on meals?"
    for item in retrieve(question, embedder, store):
        print(f"{item.score:.4f}  {item.chunk.chunk_id}  {item.chunk.section_title}")
        print(item.chunk.text)
        print()
