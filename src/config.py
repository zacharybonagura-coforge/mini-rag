import os
from dataclasses import dataclass

from enums import EmbeddingProvider, GenerationProvider, StoreProvider


@dataclass(frozen=True)
class Settings:
    embedding_provider: EmbeddingProvider
    embedding_model: str
    embedding_dimension: int
    generation_provider: GenerationProvider
    generation_model: str
    store_provider: StoreProvider
    ollama_host: str
    database_url: str
    policy_path: str
    retrieve_k: int


def load_settings() -> Settings:
    return Settings(
        embedding_provider=EmbeddingProvider(
            os.getenv("EMBEDDING_PROVIDER", EmbeddingProvider.OLLAMA.value)
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        generation_provider=GenerationProvider(
            os.getenv("GENERATION_PROVIDER", GenerationProvider.OLLAMA.value)
        ),
        generation_model=os.getenv("GENERATION_MODEL", "mistral:7b"),
        store_provider=StoreProvider(
            os.getenv("STORE_PROVIDER", StoreProvider.PGVECTOR.value)
        ),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql://rag:rag@localhost:5432/mini_rag",
        ),
        policy_path=os.getenv("POLICY_PATH", "policy.md"),
        retrieve_k=int(os.getenv("RETRIEVE_K", "3")),
    )