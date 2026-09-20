import asyncio
import os
from uuid import uuid4

import pytest

from mini_rag_lab.adapters.pgvector import PgVectorChunkRepository, create_pool
from mini_rag_lab.config import get_settings
from mini_rag_lab.domain.models import EmbeddedChunk
from mini_rag_lab.migrations import apply_migrations

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="set RUN_INTEGRATION_TESTS=1 to use PostgreSQL",
    ),
]


def _chunk(
    chunk_id: str,
    document: str,
    version: str,
    section: str,
    embedding: list[float],
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document=document,
        version=version,
        section=section,
        section_title=f"Section {section}",
        text=f"Text for section {section}",
        embedding=embedding,
        embedding_model="integration-test",
    )


async def _run_repository_test() -> None:
    settings = get_settings()
    await apply_migrations(settings.database_url)
    assert await apply_migrations(settings.database_url) == []

    token = uuid4().hex
    document = f"Integration Policy {token}"
    version = "1.0"
    first = _chunk(
        f"integration:{token}:1",
        document,
        version,
        "1",
        [1.0] + [0.0] * 767,
    )
    second = _chunk(
        f"integration:{token}:2",
        document,
        version,
        "2",
        [0.0, 1.0] + [0.0] * 766,
    )

    pool = create_pool(settings.database_url)
    async with pool:
        repository = PgVectorChunkRepository(pool)
        try:
            async with pool.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT COUNT(*) FROM schema_migrations
                    WHERE version = %s
                    """,
                    ("001_create_policy_chunks.sql",),
                )
                assert await cursor.fetchone() == (1,)

            await repository.replace_document([first, second])
            await repository.replace_document([first, second])

            async with pool.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT COUNT(*), MIN(vector_dims(embedding)),
                           MAX(vector_dims(embedding))
                    FROM policy_chunks
                    WHERE document = %s AND version = %s
                    """,
                    (document, version),
                )
                assert await cursor.fetchone() == (2, 768, 768)

            results = await repository.search(first.embedding, limit=3)
            assert len(results) <= 3
            assert [result.distance for result in results] == sorted(
                result.distance for result in results
            )
            assert results[0].chunk_id == first.chunk_id

            await repository.replace_document([first])
            async with pool.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT COUNT(*) FROM policy_chunks
                    WHERE document = %s AND version = %s
                    """,
                    (document, version),
                )
                assert await cursor.fetchone() == (1,)
        finally:
            async with pool.connection() as connection:
                await connection.execute(
                    "DELETE FROM policy_chunks WHERE document = %s",
                    (document,),
                )


def test_pgvector_repository_is_transactional_and_idempotent() -> None:
    asyncio.run(_run_repository_test())
