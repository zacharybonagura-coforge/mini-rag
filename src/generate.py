
from embeddings.base import Chunk, EmbeddingAdapter, ScoredChunk
from generation.base import ModelAdapter
from response import Citation, RagResponse, RetrievedChunkRef
from retrieve import retrieve
from store.base import VectorStoreAdapter

INSTRUCTION = """\
You are given a policy excerpt. Apply that excerpt to the question.
State the actual rule (dollar amounts, approval, class of travel, what is not reimbursable).
Compare numbers when the question includes an amount.
Answer in one or two sentences and cite the section.
Do not refuse when the excerpt decides the question.
Do not say something is allowed unless the excerpt says so.
If both the question and the excerpt include dollar amounts, first say which number is larger, then apply the rule.
If the excerpt never mentions the subject, respond with exactly:
"The provided policy does not answer this question."
Use the insufficient sentence only as the entire answer, and only when the excerpt cannot decide the question.
Never append it after a real answer.
"""

INSUFFICIENT_MARKERS = (
    "does not answer",
    "does not provide",
    "does not contain",
)


def section_label(chunk: Chunk) -> str:
    return f"{chunk.section}. {chunk.section_title}"


def build_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    excerpts = []
    for item in chunks:
        section = item.chunk
        excerpts.append(
            f"Section {section_label(section)}\n{section.text}"
        )
    joined = "\n\n".join(excerpts)
    return (
        f"{INSTRUCTION}\n"
        f"Question:\n{question}\n\n"
        f"Policy excerpts:\n{joined}"
    )


def generate_answer(
    question: str,
    chunks: list[ScoredChunk],
    adapter: ModelAdapter,
) -> str:
    return adapter.generate(build_prompt(question, chunks))


def is_insufficient(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in INSUFFICIENT_MARKERS)


def to_response(answer: str, chunks: list[ScoredChunk]) -> RagResponse:
    retrieved = [
        RetrievedChunkRef(section=section_label(item.chunk), distance=item.score)
        for item in chunks
    ]
    citation = None
    if chunks and not is_insufficient(answer.strip()):
        top = chunks[0].chunk
        citation = Citation(
            document=top.document,
            version=top.version,
            section=section_label(top),
        )
    return RagResponse(
        answer=answer,
        citation=citation,
        retrieved_chunks=retrieved,
    )


def generate(
    question: str,
    embedder: EmbeddingAdapter,
    generator: ModelAdapter,
    store: VectorStoreAdapter,
    k: int
) -> RagResponse:
    retrieved = retrieve(question, embedder, store, k=k)
    answer = generator.generate(build_prompt(question, retrieved))
    return to_response(answer, retrieved)