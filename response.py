from pydantic import BaseModel


class Citation(BaseModel):
    document: str
    version: str
    section: str


class RetrievedChunkRef(BaseModel):
    section: str
    distance: float


class RagResponse(BaseModel):
    answer: str
    citation: Citation | None
    retrieved_chunks: list[RetrievedChunkRef]
