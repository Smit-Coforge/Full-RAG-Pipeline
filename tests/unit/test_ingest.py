import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from mini_rag_lab.domain.models import EmbeddedChunk
from mini_rag_lab.services.ingestion import IngestionError, ingest_policy

# ingest_policy still calls parse_policy, which accepts this markdown shape
# and rejects any other section count. The fixture is not policy.md.
SECTION_COUNT = 6


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


def _markdown_policy() -> str:
    sections = [
        f"## {number}. Section {number}\nBody of section {number}."
        for number in range(1, SECTION_COUNT + 1)
    ]
    return "# Sample Policy — Version 1.0\n\n" + "\n\n".join(sections) + "\n"


def _write_policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.md"
    path.write_text(_markdown_policy(), encoding="utf-8")
    return path


def test_ingestion_embeds_and_saves_each_section(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(
        [[float(index)] * 768 for index in range(SECTION_COUNT)]
    )
    repository = RecordingRepository()

    chunks = asyncio.run(
        ingest_policy(
            _write_policy(tmp_path),
            provider,
            repository,
            embedding_model="test-model",
            embedding_dimensions=768,
        )
    )

    assert len(provider.received_texts) == SECTION_COUNT
    assert len(chunks) == SECTION_COUNT
    assert repository.saved == chunks
    assert all(chunk.embedding_model == "test-model" for chunk in chunks)
    assert {chunk.document for chunk in chunks} == {"Sample Policy"}
    assert {chunk.version for chunk in chunks} == {"1.0"}


@pytest.mark.parametrize(
    "embeddings",
    [
        [[0.0] * 768] * (SECTION_COUNT - 1),
        [[0.0] * 767] * SECTION_COUNT,
    ],
)
def test_invalid_embedding_response_does_not_write(
    tmp_path: Path,
    embeddings: list[list[float]],
) -> None:
    provider = FakeEmbeddingProvider(embeddings)
    repository = RecordingRepository()

    with pytest.raises(IngestionError):
        asyncio.run(
            ingest_policy(
                _write_policy(tmp_path),
                provider,
                repository,
                embedding_model="test-model",
                embedding_dimensions=768,
            )
        )

    assert repository.saved is None
