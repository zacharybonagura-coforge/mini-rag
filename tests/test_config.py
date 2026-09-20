import pytest

from config import load_settings
from enums import EmbeddingProvider, StoreProvider


def test_default_providers(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("STORE_PROVIDER", raising=False)
    settings = load_settings()
    assert settings.embedding_provider is EmbeddingProvider.OLLAMA
    assert settings.store_provider is StoreProvider.PGVECTOR
    assert settings.retrieve_k == 3
    assert settings.output_path == "runs/eval-run.json"


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "nope")
    with pytest.raises(ValueError):
        load_settings()
