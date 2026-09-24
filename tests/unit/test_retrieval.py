import asyncio
from collections.abc import Sequence

import pytest

from mini_rag_lab.adapters.pgvector import keyword_terms
from mini_rag_lab.domain.models import RetrievedChunk
from mini_rag_lab.services.retrieval import (
    CANDIDATE_N,
    RetrievalError,
    reciprocal_rank_fusion,
    retrieve_chunks,
)


def _chunk(
    chunk_id: str,
    *,
    section: str = "1",
    text: str = "Evidence",
    distance: float = 0.1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="Policy",
        version="1.0",
        section=section,
        section_title=f"Section {section}",
        text=text,
        distance=distance,
    )


class FakeEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.calls = 0

    async def embed(
        self,
        texts: Sequence[str],
        *,
        task: str = "search_document",
    ) -> list[list[float]]:
        self.calls += 1
        self.task = task
        return self.embeddings


class RecordingRepository:
    def __init__(
        self,
        *,
        vector: list[RetrievedChunk] | None = None,
        keyword: list[RetrievedChunk] | None = None,
    ) -> None:
        self.vector = vector or [_chunk("vector-1")]
        self.keyword = keyword or []
        self.search_limit: int | None = None
        self.keyword_limit: int | None = None
        self.keyword_question: str | None = None

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        assert len(embedding) == 768
        self.search_limit = limit
        return self.vector[:limit]

    async def keyword_search(
        self,
        question: str,
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        self.keyword_question = question
        self.keyword_limit = limit
        return self.keyword[:limit]


class FakeReranker:
    """Keeps input order and stamps descending fake rerank scores."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_candidate_count: int | None = None

    def rerank(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        self.last_candidate_count = len(chunks)
        return [
            chunk.model_copy(update={"rerank_score": float(100 - index)})
            for index, chunk in enumerate(chunks[:limit])
        ]


def test_keyword_terms_keep_codes_and_drop_question_filler() -> None:
    assert keyword_terms("What does section 7.1 say about the refrigerator?") == [
        "7.1",
        "refrigerator",
    ]


def test_hybrid_retrieval_requests_candidate_pool_then_reranks() -> None:
    repository = RecordingRepository()
    provider = FakeEmbeddingProvider([[0.0] * 768])
    reranker = FakeReranker()

    chunks = asyncio.run(
        retrieve_chunks(
            "question about meals",
            provider,
            repository,
            embedding_dimensions=768,
            strategy="hybrid",
            reranker=reranker,
        )
    )

    assert [chunk.chunk_id for chunk in chunks] == ["vector-1"]
    assert chunks[0].rerank_score == 100.0
    assert repository.search_limit == CANDIDATE_N
    assert repository.keyword_limit == CANDIDATE_N
    assert provider.calls == 1
    assert provider.task == "search_query"
    assert reranker.calls == 1


def test_keyword_only_skips_embedding() -> None:
    repository = RecordingRepository(keyword=[_chunk("kw-1", distance=0.0)])
    provider = FakeEmbeddingProvider([[0.0] * 768])

    chunks = asyncio.run(
        retrieve_chunks(
            "7.1 refrigerator",
            provider,
            repository,
            embedding_dimensions=768,
            strategy="keyword",
        )
    )

    assert chunks == [_chunk("kw-1", distance=0.0)]
    assert repository.search_limit is None
    assert provider.calls == 0


def test_rrf_includes_keyword_only_chunk_in_fused_candidates() -> None:
    keyword_hit = _chunk(
        "hr-policy:v2.0:section-7",
        section="7",
        text="7.1 Shared Refrigerator Policy requires labels.",
        distance=0.0,
    )
    vector_only = [
        _chunk("a", section="1", distance=0.2),
        _chunk("b", section="2", distance=0.3),
        _chunk("c", section="3", distance=0.4),
    ]

    fused = reciprocal_rank_fusion([keyword_hit], vector_only, limit=8)

    assert keyword_hit.chunk_id in {chunk.chunk_id for chunk in fused}
    assert {chunk.chunk_id for chunk in fused} == {
        keyword_hit.chunk_id,
        "a",
        "b",
        "c",
    }


def test_hybrid_retrieve_includes_keyword_miss_from_vector() -> None:
    keyword_hit = _chunk(
        "hr-policy:v2.0:section-7",
        section="7",
        text="7.1 Shared Refrigerator Policy requires labels.",
        distance=0.0,
    )
    vector_only = [
        _chunk("a", section="1", distance=0.2),
        _chunk("b", section="2", distance=0.3),
        _chunk("c", section="3", distance=0.4),
    ]
    repository = RecordingRepository(vector=vector_only, keyword=[keyword_hit])
    provider = FakeEmbeddingProvider([[0.0] * 768])
    reranker = FakeReranker()
    question = "What does section 7.1 say about the refrigerator?"

    vector_chunks = asyncio.run(
        retrieve_chunks(
            question,
            provider,
            repository,
            embedding_dimensions=768,
            strategy="vector",
            reranker=reranker,
        )
    )
    hybrid_chunks = asyncio.run(
        retrieve_chunks(
            question,
            provider,
            repository,
            embedding_dimensions=768,
            strategy="hybrid",
            reranker=reranker,
        )
    )

    expected_id = keyword_hit.chunk_id
    assert expected_id not in {chunk.chunk_id for chunk in vector_chunks}
    assert expected_id in {chunk.chunk_id for chunk in hybrid_chunks}
    assert reranker.last_candidate_count == 4


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
                strategy="vector",
            )
        )
