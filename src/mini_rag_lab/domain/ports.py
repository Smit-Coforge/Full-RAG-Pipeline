from collections.abc import Sequence
from typing import Protocol

from mini_rag_lab.domain.models import (
    EmbeddedChunk,
    GenerationDecision,
    RetrievedChunk,
)


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class AnswerGenerator(Protocol):
    async def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> GenerationDecision: ...


class ChunkRepository(Protocol):
    async def replace_document(self, chunks: Sequence[EmbeddedChunk]) -> None: ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[RetrievedChunk]: ...
