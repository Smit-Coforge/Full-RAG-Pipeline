import asyncio
import os

import pytest

from mini_rag_lab.adapters.ollama import OllamaAnswerGenerator, OllamaEmbeddingProvider
from mini_rag_lab.adapters.pgvector import PgVectorChunkRepository, create_pool
from mini_rag_lab.config import get_settings
from mini_rag_lab.domain.models import REFUSAL_ANSWER
from mini_rag_lab.services.ingestion import ingest_policy
from mini_rag_lab.services.query import GroundedQueryService

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1",
        reason="set RUN_LIVE_TESTS=1 to use PostgreSQL and Ollama",
    ),
]

CASES = [
    ("How much can I spend on food each day?", "1. Meals"),
    ("Can I book first-class airfare?", "3. Airfare"),
    ("My hotel costs $250. What do I need?", "2. Hotels"),
    ("Do I need a receipt for a $20 taxi?", "5. Receipts"),
    ("Can I claim a limousine upgrade?", "4. Ground Transportation"),
    ("Does the company reimburse gym memberships?", None),
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
        for _ in range(2):
            chunks = await ingest_policy(
                "policy.md",
                embedding_provider,
                repository,
                embedding_model=settings.embedding_model,
                embedding_dimensions=settings.embedding_dimensions,
            )
            assert len(chunks) == 6

        async with pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT COUNT(*), MIN(vector_dims(embedding)),
                       MAX(vector_dims(embedding))
                FROM policy_chunks
                WHERE document = %s AND version = %s
                """,
                ("Employee Expense Policy", "2.0"),
            )
            assert await cursor.fetchone() == (6, 768, 768)

        service = GroundedQueryService(
            embedding_provider,
            repository,
            OllamaAnswerGenerator(
                settings.ollama_host,
                settings.generation_model,
            ),
            embedding_dimensions=settings.embedding_dimensions,
            max_cosine_distance=settings.max_cosine_distance,
        )

        responses = {}
        for question, expected_section in CASES:
            response = await service.ask(question)
            responses[expected_section] = response
            distances = [item.distance for item in response.retrieved_chunks]
            assert len(distances) <= 3
            assert distances == sorted(distances)

            if expected_section is None:
                assert response.answer == REFUSAL_ANSWER
                assert response.citation is None
            else:
                assert response.citation is not None
                assert response.citation.section == expected_section
                assert expected_section in [
                    item.section for item in response.retrieved_chunks
                ]

        assert "$65" in responses["1. Meals"].answer
        airfare = responses["3. Airfare"].answer.lower()
        assert all(term in airfare for term in ("economy", "business", "approval"))
        hotel = responses["2. Hotels"].answer.lower()
        assert all(term in hotel for term in ("manager", "approve", "before booking"))
        receipt = responses["5. Receipts"].answer.lower()
        assert receipt.startswith("no") and "$25" in receipt
        assert (
            "not reimbursable" in responses["4. Ground Transportation"].answer.lower()
        )


def test_all_six_assignment_questions() -> None:
    asyncio.run(_run_live_pipeline())
