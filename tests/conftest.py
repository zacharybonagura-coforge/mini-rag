from pathlib import Path

import pytest

from embeddings.base import Chunk, ScoredChunk

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy_path() -> Path:
    return ROOT / "policy.md"


def make_chunk(
    section: str = "1",
    title: str = "Meals",
    text: str = "Employees may claim up to $65 per day.",
    embedding: list[float] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=text,
        embedding=embedding or [0.1, 0.2, 0.3],
    )


def make_scored(chunk: Chunk, distance: float) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=distance)
