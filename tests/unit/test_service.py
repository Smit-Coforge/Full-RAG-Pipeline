import asyncio
from collections.abc import Sequence

from mini_rag_lab.domain.models import (
    REFUSAL_ANSWER,
    GenerationDecision,
    RetrievedChunk,
)
from mini_rag_lab.services.query import GroundedQueryService


def _chunk(number: int, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{number}",
        document="Policy",
        version="1.0",
        section=str(number),
        section_title=f"Section {number}",
        text=f"Evidence {number}",
        distance=distance,
    )


class FakeEmbeddingProvider:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]


class FakeRepository:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        assert limit == 3
        return self.chunks


class FakeGenerator:
    def __init__(self, decision: GenerationDecision) -> None:
        self.decision = decision
        self.received_chunks: list[RetrievedChunk] | None = None

    async def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> GenerationDecision:
        self.received_chunks = list(chunks)
        return self.decision


def _service(
    chunks: list[RetrievedChunk],
    generator: FakeGenerator,
) -> GroundedQueryService:
    return GroundedQueryService(
        FakeEmbeddingProvider(),
        FakeRepository(chunks),
        generator,
        embedding_dimensions=768,
        max_cosine_distance=0.4,
    )


def test_supported_answer_uses_stored_citation_and_top_evidence_only() -> None:
    chunks = [_chunk(1, 0.1), _chunk(2, 0.2), _chunk(3, 0.3)]
    generator = FakeGenerator(
        GenerationDecision(answer="Grounded answer", supporting_chunk_id="chunk-1")
    )

    response = asyncio.run(_service(chunks, generator).ask("question"))

    assert response.answer == "Grounded answer"
    assert response.citation is not None
    assert response.citation.model_dump() == {
        "document": "Policy",
        "version": "1.0",
        "section": "1. Section 1",
    }
    assert [item.distance for item in response.retrieved_chunks] == [0.1, 0.2, 0.3]
    assert generator.received_chunks == [chunks[0]]


def test_invalid_model_citation_is_replaced_with_refusal() -> None:
    chunks = [_chunk(1, 0.1), _chunk(2, 0.2)]
    generator = FakeGenerator(
        GenerationDecision(answer="Unsupported", supporting_chunk_id="chunk-2")
    )

    response = asyncio.run(_service(chunks, generator).ask("question"))

    assert response.answer == REFUSAL_ANSWER
    assert response.citation is None


def test_weak_retrieval_refuses_without_calling_generator() -> None:
    chunks = [_chunk(1, 0.41)]
    generator = FakeGenerator(
        GenerationDecision(answer="Should not run", supporting_chunk_id="chunk-1")
    )

    response = asyncio.run(_service(chunks, generator).ask("question"))

    assert response.answer == REFUSAL_ANSWER
    assert response.citation is None
    assert generator.received_chunks is None
