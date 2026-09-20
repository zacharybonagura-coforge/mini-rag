from generate import is_insufficient, to_response
from tests.conftest import make_chunk, make_scored


def test_supported_answer_gets_citation():
    chunks = [make_scored(make_chunk(), 0.1)]
    response = to_response(
        "Employees may claim up to $65 per day for meals. (Section 1. Meals)",
        chunks,
    )
    assert response.citation is not None
    assert response.citation.section == "1. Meals"
    assert response.citation.document == "Employee Expense Policy"
    assert response.retrieved_chunks[0].distance == 0.1


def test_insufficient_answer_has_null_citation():
    chunks = [make_scored(make_chunk(section="4", title="Ground Transportation"), 0.3)]
    answer = "The provided policy does not answer this question."
    assert is_insufficient(answer)
    response = to_response(answer, chunks)
    assert response.citation is None
    assert len(response.retrieved_chunks) == 1
