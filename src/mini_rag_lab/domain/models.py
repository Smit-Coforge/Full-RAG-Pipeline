from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

REFUSAL_ANSWER = "The provided policy does not answer this question."
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Embedding = Annotated[list[float], Field(min_length=768, max_length=768)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicyChunk(StrictModel):
    chunk_id: NonEmptyString
    document: NonEmptyString
    version: NonEmptyString
    section: NonEmptyString
    section_title: NonEmptyString
    text: NonEmptyString


class EmbeddedChunk(PolicyChunk):
    embedding: Embedding
    embedding_model: NonEmptyString


class RetrievedChunk(PolicyChunk):
    distance: float


class AskRequest(StrictModel):
    question: NonEmptyString


class Citation(StrictModel):
    document: NonEmptyString
    version: NonEmptyString
    section: NonEmptyString


class RetrievedChunkSummary(StrictModel):
    section: NonEmptyString
    distance: float


class GenerationDecision(StrictModel):
    answer: NonEmptyString
    supporting_chunk_id: NonEmptyString | None = None


class AskResponse(StrictModel):
    answer: NonEmptyString
    citation: Citation | None
    retrieved_chunks: list[RetrievedChunkSummary] = Field(max_length=3)
