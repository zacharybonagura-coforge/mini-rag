from ingest import ingest
from tests.test_retrieve import FakeEmbedder, FakeStore  # or from conftest


class RecordingStore(FakeStore):
    def __init__(self):
        super().__init__([])
        self.saved = []

    def save_chunks(self, chunks):
        self.saved = list(chunks)


def test_ingest_writes_six_embedded_chunks(policy_path):
    store = RecordingStore()
    chunks = ingest(policy_path, FakeEmbedder(), store)
    assert len(chunks) == 6
    assert len(store.saved) == 6
    assert all(len(c.embedding) == 3 for c in chunks)
    assert chunks[0].chunk_id == "expense-policy:v2.0:section-1"
