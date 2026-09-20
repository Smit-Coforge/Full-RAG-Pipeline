import asyncio
from collections.abc import Sequence

import pytest

from mini_rag_lab.domain.models import RetrievedChunk
from mini_rag_lab.services.retrieval import RetrievalError, retrieve_chunks


def _chunk(distance: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document="Policy",
        version="1.0",
        section="1",
        section_title="Test",
        text="Evidence",
        distance=distance,
    )


class FakeEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        assert texts == ["question"]
        return self.embeddings


class RecordingRepository:
    def __init__(self) -> None:
        self.limit: int | None = None

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        assert len(embedding) == 768
        self.limit = limit
        return [_chunk()]


def test_retrieval_requests_exactly_three_results() -> None:
    repository = RecordingRepository()

    chunks = asyncio.run(
        retrieve_chunks(
            "question",
            FakeEmbeddingProvider([[0.0] * 768]),
            repository,
            embedding_dimensions=768,
        )
    )

    assert chunks == [_chunk()]
    assert repository.limit == 3


@pytest.mark.parametrize(
    "embeddings",
    [
        [],
        [[0.0] * 768, [0.0] * 768],
        [[0.0] * 767],
    ],
)
def test_retrieval_rejects_invalid_question_embeddings(
    embeddings: list[list[float]],
) -> None:
    with pytest.raises(RetrievalError):
        asyncio.run(
            retrieve_chunks(
                "question",
                FakeEmbeddingProvider(embeddings),
                RecordingRepository(),
                embedding_dimensions=768,
            )
        )
