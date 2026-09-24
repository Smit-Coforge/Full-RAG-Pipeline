import asyncio
import os

import pytest

from mini_rag_lab.adapters.cross_encoder import CrossEncoderReranker
from mini_rag_lab.adapters.ollama import OllamaAnswerGenerator, OllamaEmbeddingProvider
from mini_rag_lab.adapters.pgvector import PgVectorChunkRepository, create_pool
from mini_rag_lab.config import get_settings
from mini_rag_lab.services.ingestion import ingest_corpus
from mini_rag_lab.services.query import GroundedQueryService

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1",
        reason="set RUN_LIVE_TESTS=1 to use PostgreSQL and Ollama",
    ),
]


async def _run_live_pipeline() -> None:
    settings = get_settings()
    pool = create_pool(settings.database_url)
    embedding_provider = OllamaEmbeddingProvider(
        settings.ollama_host,
        settings.embedding_model,
    )

    async with pool:
        repository = PgVectorChunkRepository(pool)
        chunks = await ingest_corpus(
            "corpus",
            embedding_provider,
            repository,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        )
        assert len(chunks) >= 1

        async with pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT COUNT(*), MIN(vector_dims(embedding)),
                       MAX(vector_dims(embedding))
                FROM policy_chunks
                """
            )
            count, min_dims, max_dims = await cursor.fetchone()
            assert count == len(chunks)
            assert min_dims == max_dims == 768

        service = GroundedQueryService(
            embedding_provider,
            repository,
            OllamaAnswerGenerator(
                settings.ollama_host,
                settings.generation_model,
            ),
            embedding_dimensions=settings.embedding_dimensions,
            max_cosine_distance=settings.max_cosine_distance,
            reranker=CrossEncoderReranker(settings.reranker_model),
        )

        question = "What does section 7.1 say about the refrigerator?"
        response = await service.ask(question, strategy="hybrid")
        assert response.retrieval_strategy == "hybrid"
        assert response.answer
        assert response.citation is not None
        assert response.retrieved_chunks
        assert len(response.retrieved_chunks) <= 3
        assert all(
            item.rerank_score is not None for item in response.retrieved_chunks
        )
        scores = [item.rerank_score for item in response.retrieved_chunks]
        assert scores == sorted(scores, reverse=True)


def test_corpus_hybrid_pipeline() -> None:
    asyncio.run(_run_live_pipeline())
