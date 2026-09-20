import httpx


class OllamaAdapter:
    provider = "ollama"

    def __init__(
        self,
        model_id: str = "mistral:7b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model_id = model_id
        self._url = f"{host}/api/generate"

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            self._url,
            json={
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
