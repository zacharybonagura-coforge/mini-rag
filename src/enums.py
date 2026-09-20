from enum import Enum


class EmbeddingProvider(str, Enum):
    OLLAMA = "ollama"


class GenerationProvider(str, Enum):
    OLLAMA = "ollama"


class StoreProvider(str, Enum):
    PGVECTOR = "pgvector"
