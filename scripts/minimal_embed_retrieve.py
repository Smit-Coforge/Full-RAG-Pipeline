"""Minimal embed → store → retrieve proof on Ollama + pgvector.

Uses two fixed sentences. A query about the first sentence must rank that
row ahead of the second. No chunking, hybrid search, or generation.
"""

from __future__ import annotations

import asyncio
import json
import sys

from ollama import AsyncClient
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from mini_rag_lab.config import get_settings

SENTENCE_A = "Employees may claim up to sixty-five dollars per day for meals."
SENTENCE_B = "Overnight hotels require a pre-approved travel request."
QUERY = "How much can I spend on food each day?"

TABLE = "minimal_embed_proof"


async def main() -> int:
    settings = get_settings()
    client = AsyncClient(host=settings.ollama_host)
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=1,
        open=False,
        configure=register_vector_async,
    )
    await pool.open()

    try:
        embed_response = await client.embed(
            model=settings.embedding_model,
            input=[
                f"search_document: {SENTENCE_A}",
                f"search_document: {SENTENCE_B}",
                f"search_query: {QUERY}",
            ],
        )
        vectors = [list(embedding) for embedding in embed_response.embeddings]
        if len(vectors) != 3:
            raise RuntimeError(f"expected 3 embeddings, received {len(vectors)}")
        for index, vector in enumerate(vectors):
            if len(vector) != settings.embedding_dimensions:
                raise RuntimeError(
                    f"embedding {index} has {len(vector)} dimensions; "
                    f"expected {settings.embedding_dimensions}"
                )

        vector_a, vector_b, query_vector = vectors
        async with pool.connection() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding VECTOR(768) NOT NULL
                )
                """
            )
            await connection.execute(f"DELETE FROM {TABLE}")
            await connection.execute(
                f"""
                INSERT INTO {TABLE} (chunk_id, text, embedding)
                VALUES
                    (%s, %s, %s),
                    (%s, %s, %s)
                """,
                (
                    "proof:a",
                    SENTENCE_A,
                    Vector(vector_a),
                    "proof:b",
                    SENTENCE_B,
                    Vector(vector_b),
                ),
            )
            cursor = await connection.execute(
                f"""
                SELECT chunk_id, text, embedding <=> %s AS distance
                FROM {TABLE}
                ORDER BY embedding <=> %s ASC
                """,
                (Vector(query_vector), Vector(query_vector)),
            )
            rows = await cursor.fetchall()

        results = [
            {
                "chunk_id": row[0],
                "text": row[1],
                "distance": float(row[2]),
            }
            for row in rows
        ]
        if not results or results[0]["chunk_id"] != "proof:a":
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "query did not rank sentence A first",
                        "results": results,
                    },
                    indent=2,
                )
            )
            return 1

        print(
            json.dumps(
                {
                    "ok": True,
                    "embedding_model": settings.embedding_model,
                    "dimensions": settings.embedding_dimensions,
                    "query": QUERY,
                    "results": results,
                },
                indent=2,
            )
        )
        return 0
    finally:
        async with pool.connection() as connection:
            await connection.execute(f"DROP TABLE IF EXISTS {TABLE}")
        await pool.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
