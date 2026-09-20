from mini_rag_lab.domain.models import RetrievedChunk
from mini_rag_lab.domain.ports import ChunkRepository, EmbeddingProvider

TOP_K = 3


class RetrievalError(RuntimeError):
    pass


async def retrieve_chunks(
    question: str,
    embedding_provider: EmbeddingProvider,
    repository: ChunkRepository,
    *,
    embedding_dimensions: int,
) -> list[RetrievedChunk]:
    embeddings = await embedding_provider.embed([question])
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

    return await repository.search(embedding, limit=TOP_K)
