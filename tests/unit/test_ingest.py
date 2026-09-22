import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from mini_rag_lab.domain.models import EmbeddedChunk, PolicyChunk
from mini_rag_lab.services.ingestion import (
    IngestionError,
    ingest_corpus,
    ingest_policy,
)

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
        self.calls: list[list[EmbeddedChunk]] = []

    async def replace_document(self, chunks: Sequence[EmbeddedChunk]) -> None:
        self.saved = list(chunks)
        self.calls.append(self.saved)


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


def _policy_chunk(version: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=f"hr-policy:v{version}:section-1",
        document="HR Policy",
        version=version,
        section="1",
        section_title="Purpose",
        text=f"Body for version {version}.",
    )


class PerTextEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]] | None = None) -> None:
        self.embeddings = embeddings
        self.received_texts: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.received_texts.append(batch)
        if self.embeddings is not None:
            return self.embeddings
        return [[0.0] * 768 for _ in batch]


def test_corpus_ingest_replaces_each_file_and_skips_other_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    (tmp_path / "hr-v1.pdf").write_bytes(b"%PDF")
    (tmp_path / "hr-v2.docx").write_bytes(b"PK")

    def load(path: Path) -> list[PolicyChunk]:
        version = "1.0" if path.suffix.lower() == ".pdf" else "2.0"
        return [_policy_chunk(version)]

    monkeypatch.setattr("mini_rag_lab.services.ingestion.load_policy_file", load)
    provider = PerTextEmbeddingProvider()
    repository = RecordingRepository()

    chunks = asyncio.run(
        ingest_corpus(
            tmp_path,
            provider,
            repository,
            embedding_model="test-model",
            embedding_dimensions=768,
        )
    )

    assert [batch[0] for batch in provider.received_texts] == [
        "Body for version 1.0.",
        "Body for version 2.0.",
    ]
    assert [call[0].version for call in repository.calls] == ["1.0", "2.0"]
    assert [chunk.version for chunk in chunks] == ["1.0", "2.0"]
    assert all(chunk.embedding_model == "test-model" for chunk in chunks)


def test_corpus_invalid_embedding_does_not_replace_that_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "hr-v1.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(
        "mini_rag_lab.services.ingestion.load_policy_file",
        lambda path: [_policy_chunk("1.0")],
    )
    provider = PerTextEmbeddingProvider(embeddings=[[0.0] * 767])
    repository = RecordingRepository()

    with pytest.raises(IngestionError):
        asyncio.run(
            ingest_corpus(
                tmp_path,
                provider,
                repository,
                embedding_model="test-model",
                embedding_dimensions=768,
            )
        )

    assert repository.calls == []
