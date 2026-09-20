from typing import Protocol


class ModelAdapter(Protocol):
    provider: str
    model_id: str

    def generate(self, prompt: str) -> str: ...
