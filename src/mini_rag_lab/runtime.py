from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from mini_rag_lab.adapters.ollama import OllamaAnswerGenerator, OllamaEmbeddingProvider
from mini_rag_lab.adapters.pgvector import PgVectorChunkRepository, create_pool
from mini_rag_lab.config import Settings, get_settings
from mini_rag_lab.services.query import GroundedQueryService


@dataclass(slots=True)
class ApplicationRuntime:
    settings: Settings
    pool: AsyncConnectionPool
    embedding_provider: OllamaEmbeddingProvider
    repository: PgVectorChunkRepository
    answer_generator: OllamaAnswerGenerator
    service: GroundedQueryService

    async def close(self) -> None:
        await self.pool.close()


async def create_runtime(settings: Settings | None = None) -> ApplicationRuntime:
    resolved_settings = settings or get_settings()
    pool = create_pool(resolved_settings.database_url)
    try:
        await pool.open(wait=True)
    except Exception:
        await pool.close()
        raise

    embedding_provider = OllamaEmbeddingProvider(
        resolved_settings.ollama_host,
        resolved_settings.embedding_model,
    )
    repository = PgVectorChunkRepository(pool)
    answer_generator = OllamaAnswerGenerator(
        resolved_settings.ollama_host,
        resolved_settings.generation_model,
        thinking=resolved_settings.generation_thinking,
    )
    service = GroundedQueryService(
        embedding_provider,
        repository,
        answer_generator,
        embedding_dimensions=resolved_settings.embedding_dimensions,
        max_cosine_distance=resolved_settings.max_cosine_distance,
    )
    return ApplicationRuntime(
        settings=resolved_settings,
        pool=pool,
        embedding_provider=embedding_provider,
        repository=repository,
        answer_generator=answer_generator,
        service=service,
    )
