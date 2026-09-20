import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    generation_provider: str
    generation_model: str
    store_provider: str
    ollama_host: str
    database_url: str | None
    policy_path: str
    retrieve_k: int


def load_settings() -> Settings:
    return Settings(
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        generation_provider=os.getenv("GENERATION_PROVIDER", "ollama"),
        generation_model=os.getenv("GENERATION_MODEL", "mistral:7b"),
        store_provider=os.getenv("STORE_PROVIDER", "pgvector"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        database_url=os.getenv("DATABASE_URL"),
        policy_path=os.getenv("POLICY_PATH", "policy.md"),
        retrieve_k=int(os.getenv("RETRIEVE_K", "1")),
    )
