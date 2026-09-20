from config import Settings
from enums import (
    EmbeddingProvider,
    GenerationProvider,
    StoreProvider
)
from embeddings.base import EmbeddingAdapter
from embeddings.ollama import OllamaEmbeddingAdapter
from generation.base import ModelAdapter
from generation.ollama import OllamaAdapter
from store.base import VectorStoreAdapter
from store.pgvector import PgVectorStoreAdapter


def build_embedder(settings: Settings) -> EmbeddingAdapter:
    if settings.embedding_provider is EmbeddingProvider.OLLAMA:
        return OllamaEmbeddingAdapter(
            model_id=settings.embedding_model,
            dimension=settings.embedding_dimension,
            host=settings.ollama_host,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def build_generator(settings: Settings) -> ModelAdapter:
    if settings.generation_provider is GenerationProvider.OLLAMA:
        return OllamaAdapter(
            model_id=settings.generation_model,
            host=settings.ollama_host,
        )
    raise ValueError(f"Unknown generation provider: {settings.generation_provider}")


def build_store(settings: Settings) -> VectorStoreAdapter:
    if settings.store_provider is StoreProvider.PGVECTOR:
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required for store_provider=pgvector")
        return PgVectorStoreAdapter(dsn=settings.database_url)
    raise ValueError(f"Unknown store provider: {settings.store_provider}")