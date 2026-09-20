import asyncio
from collections.abc import Sequence

import pytest

from mini_rag_lab.domain.models import EmbeddedChunk
from mini_rag_lab.services.ingestion import IngestionError, ingest_policy


class FakeEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.received_texts: list[str] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.received_texts = list(texts)
        return self.embeddings


class RecordingRepository:
    def __init__(self) -> None:
        self.saved: list[EmbeddedChunk] | None = None

    async def replace_document(self, chunks: Sequence[EmbeddedChunk]) -> None:
        self.saved = list(chunks)


def test_ingestion_embeds_and_saves_all_six_chunks() -> None:
    provider = FakeEmbeddingProvider([[float(index)] * 768 for index in range(6)])
    repository = RecordingRepository()

    chunks = asyncio.run(
        ingest_policy(
            "policy.md",
            provider,
            repository,
            embedding_model="test-model",
            embedding_dimensions=768,
        )
    )

    assert len(provider.received_texts) == 6
    assert len(chunks) == 6
    assert repository.saved == chunks
    assert all(chunk.embedding_model == "test-model" for chunk in chunks)


@pytest.mark.parametrize(
    "embeddings",
    [
        [[0.0] * 768] * 5,
        [[0.0] * 767] * 6,
    ],
)
def test_invalid_embedding_response_does_not_write(
    embeddings: list[list[float]],
) -> None:
    provider = FakeEmbeddingProvider(embeddings)
    repository = RecordingRepository()

    with pytest.raises(IngestionError):
        asyncio.run(
            ingest_policy(
                "policy.md",
                provider,
                repository,
                embedding_model="test-model",
                embedding_dimensions=768,
            )
        )

    assert repository.saved is None
