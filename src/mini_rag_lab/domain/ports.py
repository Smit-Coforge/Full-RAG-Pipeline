from collections.abc import Sequence
from typing import Literal, Protocol

from mini_rag_lab.domain.models import (
    EmbeddedChunk,
    GenerationDecision,
    RetrievedChunk,
)

EmbeddingTask = Literal["search_document", "search_query"]


class EmbeddingProvider(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
        *,
        task: EmbeddingTask = "search_document",
    ) -> list[list[float]]: ...


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

    async def keyword_search(
        self,
        question: str,
        *,
        limit: int,
    ) -> list[RetrievedChunk]: ...
