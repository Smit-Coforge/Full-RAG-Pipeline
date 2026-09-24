import re
from collections.abc import Sequence

from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool
from sklearn.feature_extraction import text as sklearn_text

from mini_rag_lab.domain.models import EmbeddedChunk, RetrievedChunk

_SECTION_CODE = re.compile(r"\d+\.\d+")
_KEYWORD_WORD = re.compile(r"[A-Za-z]{4,}")
_MAX_SEARCH_LIMIT = 8
# Generic English stopwords plus policy-lab terms and auxiliaries sklearn omits.
_KEYWORD_STOPWORDS = frozenset(sklearn_text.ENGLISH_STOP_WORDS) | {
    "did",
    "does",
    "doing",
    "policy",
    "section",
}


def _like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def keyword_terms(question: str) -> list[str]:
    """Section codes plus content words; skip stopwords and domain filler."""
    terms: list[str] = []
    seen: set[str] = set()
    for term in _SECTION_CODE.findall(question):
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    for term in _KEYWORD_WORD.findall(question):
        key = term.casefold()
        if key in seen or key in _KEYWORD_STOPWORDS:
            continue
        seen.add(key)
        terms.append(term)
    return terms


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

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        if not 1 <= limit <= _MAX_SEARCH_LIMIT:
            raise ValueError(
                f"retrieval limit must be between 1 and {_MAX_SEARCH_LIMIT}"
            )

        query_vector = Vector(list(embedding))
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    chunk_id,
                    document,
                    version,
                    section,
                    section_title,
                    text,
                    embedding <=> %s AS distance
                FROM policy_chunks
                ORDER BY embedding <=> %s ASC
                LIMIT %s
                """,
                (query_vector, query_vector, limit),
            )
            rows = await cursor.fetchall()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document=row[1],
                version=row[2],
                section=row[3],
                section_title=row[4],
                text=row[5],
                distance=float(row[6]),
            )
            for row in rows
        ]

    async def keyword_search(
        self,
        question: str,
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        if not 1 <= limit <= _MAX_SEARCH_LIMIT:
            raise ValueError(
                f"retrieval limit must be between 1 and {_MAX_SEARCH_LIMIT}"
            )

        terms = keyword_terms(question)
        if not terms:
            return []

        patterns = [_like_pattern(term) for term in terms]
        clauses = " OR ".join(
            "(text ILIKE %s ESCAPE '\\' OR section_title ILIKE %s ESCAPE '\\')"
            for _ in patterns
        )
        params: list[object] = []
        for pattern in patterns:
            params.extend((pattern, pattern))

        code_patterns = [
            _like_pattern(term) for term in terms if _SECTION_CODE.fullmatch(term)
        ]
        if code_patterns:
            code_rank = " OR ".join(
                "(text ILIKE %s ESCAPE '\\' OR section_title ILIKE %s ESCAPE '\\')"
                for _ in code_patterns
            )
            for pattern in code_patterns:
                params.extend((pattern, pattern))
            order_by = f"CASE WHEN ({code_rank}) THEN 0 ELSE 1 END, chunk_id"
        else:
            order_by = "chunk_id"

        params.append(limit)

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT
                    chunk_id,
                    document,
                    version,
                    section,
                    section_title,
                    text,
                    0.0 AS distance
                FROM policy_chunks
                WHERE {clauses}
                ORDER BY {order_by}
                LIMIT %s
                """,
                params,
            )
            rows = await cursor.fetchall()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document=row[1],
                version=row[2],
                section=row[3],
                section_title=row[4],
                text=row[5],
                distance=float(row[6]),
            )
            for row in rows
        ]
