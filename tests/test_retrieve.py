from retrieve import retrieve
from tests.conftest import make_chunk, make_scored


class FakeEmbedder:
    provider = "fake"
    model_id = "fake"
    dimension = 3

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[0.0, 1.0, 0.0] for _ in texts]


class FakeStore:
    provider = "fake"

    def __init__(self, scored):
        self._scored = scored

    def save_chunks(self, chunks): ...
    def load_chunks(self):
        return []

    def search(self, query_embedding, k=3):
        return self._scored[:k]


def test_retrieve_respects_k_and_order():
    scored = [
        make_scored(make_chunk(section="1", title="Meals"), 0.1),
        make_scored(make_chunk(section="2", title="Hotels"), 0.2),
        make_scored(make_chunk(section="3", title="Airfare"), 0.3),
    ]
    results = retrieve("food budget", FakeEmbedder(), FakeStore(scored), k=2)
    assert len(results) == 2
    assert results[0].score <= results[1].score
    assert results[0].chunk.section == "1"
