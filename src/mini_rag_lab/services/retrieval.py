import asyncio
from collections.abc import Sequence
from typing import Literal

from mini_rag_lab.domain.models import RetrievedChunk
from mini_rag_lab.domain.ports import ChunkRepository, EmbeddingProvider

CANDIDATE_N = 8
TOP_K = 3
RetrievalStrategy = Literal["vector", "keyword", "hybrid"]


class RetrievalError(RuntimeError):
    pass


def merge_retrieval_results(
    keyword_chunks: Sequence[RetrievedChunk],
    vector_chunks: Sequence[RetrievedChunk],
    *,
    limit: int = TOP_K,
) -> list[RetrievedChunk]:
    """Union by chunk_id. Keyword hits first so exact matches stay in top-k."""
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in (*keyword_chunks, *vector_chunks):
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged


async def _vector_search(
    question: str,
    embedding_provider: EmbeddingProvider,
    repository: ChunkRepository,
    *,
    embedding_dimensions: int,
) -> list[RetrievedChunk]:
    embeddings = await embedding_provider.embed([question], task="search_query")
    if len(embeddings) != 1:
        raise RetrievalError(
            f"expected one question embedding, received {len(embeddings)}"
        )

    embedding = embeddings[0]
    if len(embedding) != embedding_dimensions:
        raise RetrievalError(
            f"question embedding has {len(embedding)} dimensions; "
            f"expected {embedding_dimensions}"
        )

    return await repository.search(embedding, limit=CANDIDATE_N)


async def retrieve_chunks(
    question: str,
    embedding_provider: EmbeddingProvider,
    repository: ChunkRepository,
    *,
    embedding_dimensions: int,
    strategy: RetrievalStrategy = "hybrid",
) -> list[RetrievedChunk]:
    if strategy == "keyword":
        keyword_chunks = await repository.keyword_search(
            question,
            limit=CANDIDATE_N,
        )
        return keyword_chunks[:TOP_K]

    if strategy == "vector":
        vector_chunks = await _vector_search(
            question,
            embedding_provider,
            repository,
            embedding_dimensions=embedding_dimensions,
        )
        return vector_chunks[:TOP_K]

    keyword_chunks, vector_chunks = await asyncio.gather(
        repository.keyword_search(question, limit=CANDIDATE_N),
        _vector_search(
            question,
            embedding_provider,
            repository,
            embedding_dimensions=embedding_dimensions,
        ),
    )
    return merge_retrieval_results(keyword_chunks, vector_chunks, limit=TOP_K)
