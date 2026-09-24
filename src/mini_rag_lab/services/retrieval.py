import asyncio
from collections.abc import Sequence
from typing import Literal

from mini_rag_lab.domain.models import RetrievedChunk
from mini_rag_lab.domain.ports import ChunkRepository, EmbeddingProvider, Reranker

CANDIDATE_N = 8
TOP_K = 3
RRF_K = 60
RetrievalStrategy = Literal["vector", "keyword", "hybrid"]


class RetrievalError(RuntimeError):
    pass


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[RetrievedChunk],
    limit: int = CANDIDATE_N,
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Merge ranked lists by Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = by_id.get(chunk.chunk_id)
            if existing is None:
                by_id[chunk.chunk_id] = chunk
            elif existing.distance == 0.0 and chunk.distance > 0.0:
                # Prefer real cosine distance when the same id appears in both.
                by_id[chunk.chunk_id] = chunk
            elif (
                chunk.distance > 0.0
                and existing.distance > 0.0
                and chunk.distance < existing.distance
            ):
                by_id[chunk.chunk_id] = chunk

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [by_id[chunk_id] for chunk_id in ordered_ids[:limit]]


# Back-compat name used in older tests/docs.
merge_retrieval_results = reciprocal_rank_fusion


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


async def _apply_rerank(
    question: str,
    candidates: Sequence[RetrievedChunk],
    reranker: Reranker,
) -> list[RetrievedChunk]:
    def _run() -> list[RetrievedChunk]:
        return reranker.rerank(question, list(candidates), limit=TOP_K)

    return await asyncio.to_thread(_run)


async def retrieve_chunks(
    question: str,
    embedding_provider: EmbeddingProvider,
    repository: ChunkRepository,
    *,
    embedding_dimensions: int,
    strategy: RetrievalStrategy = "hybrid",
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    if strategy == "keyword":
        candidates = await repository.keyword_search(question, limit=CANDIDATE_N)
    elif strategy == "vector":
        candidates = await _vector_search(
            question,
            embedding_provider,
            repository,
            embedding_dimensions=embedding_dimensions,
        )
    else:
        keyword_chunks, vector_chunks = await asyncio.gather(
            repository.keyword_search(question, limit=CANDIDATE_N),
            _vector_search(
                question,
                embedding_provider,
                repository,
                embedding_dimensions=embedding_dimensions,
            ),
        )
        candidates = reciprocal_rank_fusion(
            keyword_chunks,
            vector_chunks,
            limit=CANDIDATE_N,
        )

    if not candidates:
        return []
    if reranker is None:
        return list(candidates)[:TOP_K]
    return await _apply_rerank(question, candidates, reranker)
