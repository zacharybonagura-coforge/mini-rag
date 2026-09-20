from generate import build_prompt, generate
from tests.conftest import make_chunk, make_scored
from tests.test_retrieve import FakeEmbedder, FakeStore


class FakeGenerator:
    provider = "fake"
    model_id = "fake"

    def generate(self, prompt: str) -> str:
        return "Employees may claim up to $65 per day. (Section 1. Meals)"


def test_build_prompt_includes_question_and_section():
    chunks = [make_scored(make_chunk(), 0.1)]
    prompt = build_prompt("How much for food?", chunks)
    assert "How much for food?" in prompt
    assert "Section 1. Meals" in prompt
    assert "$65" in prompt


def test_generate_returns_structured_response():
    scored = [make_scored(make_chunk(), 0.1)]
    response = generate(
        "How much can I spend on food each day?",
        FakeEmbedder(),
        FakeGenerator(),
        FakeStore(scored),
        k=1,
    )
    assert response.citation is not None
    assert response.citation.section == "1. Meals"
    assert response.retrieved_chunks[0].distance == 0.1
