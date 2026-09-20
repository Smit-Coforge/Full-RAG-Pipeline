from collections.abc import Sequence

from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from mini_rag_lab.models import EmbeddedChunk


def create_pool(database_url: str) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=4,
        open=False,
        configure=register_vector_async,
    )


class PgVectorChunkRepository:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def replace_document(self, chunks: Sequence[EmbeddedChunk]) -> None:
        if not chunks:
            raise ValueError("at least one chunk is required")

        document = chunks[0].document
        version = chunks[0].version
        if any(
            chunk.document != document or chunk.version != version for chunk in chunks
        ):
            raise ValueError("all chunks must belong to one document version")

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        rows = [
            (
                chunk.chunk_id,
                chunk.document,
                chunk.version,
                chunk.section,
                chunk.section_title,
                chunk.text,
                Vector(chunk.embedding),
                chunk.embedding_model,
            )
            for chunk in chunks
        ]

        async with (
            self._pool.connection() as connection,
            connection.transaction(),
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                """
                DELETE FROM policy_chunks
                WHERE document = %s
                  AND version = %s
                  AND NOT (chunk_id = ANY(%s))
                """,
                (document, version, chunk_ids),
            )
            await cursor.executemany(
                """
                INSERT INTO policy_chunks (
                    chunk_id,
                    document,
                    version,
                    section,
                    section_title,
                    text,
                    embedding,
                    embedding_model
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document = EXCLUDED.document,
                    version = EXCLUDED.version,
                    section = EXCLUDED.section,
                    section_title = EXCLUDED.section_title,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                rows,
            )
