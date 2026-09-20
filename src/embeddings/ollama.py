from collections.abc import Sequence

import httpx


class OllamaEmbeddingAdapter:
    provider = "ollama"

    def __init__(
        self,
        model_id: str = "nomic-embed-text",
        dimension: int = 768,
        host: str = "http://localhost:11434",
        query_prefix: str = "search_query: ",
        document_prefix: str = "search_document: ",
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._url = f"{host}/api/embed"

    def embed_query(self, query: str) -> list[float]:
        return self._embed([f"{self.query_prefix}{query}"])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts: return []
        prefixed = [f"{self.document_prefix}{text}" for text in texts]
        return self._embed(prefixed)

    def _embed(self, inputs: Sequence[str]) -> list[list[float]]:
        response = httpx.post(
            self._url,
            json={"model": self.model_id, "input": list(inputs)},
            timeout=60.0,
        )
        response.raise_for_status()
        vectors = [list(vector) for vector in response.json()["embeddings"]]
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Expected embedding dimension {self.dimension}, got {len(vector)}"
                )
        return vectors
