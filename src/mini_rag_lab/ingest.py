from pathlib import Path

from mini_rag_lab.chunking import parse_policy
from mini_rag_lab.models import EmbeddedChunk
from mini_rag_lab.ports import ChunkRepository, EmbeddingProvider


class IngestionError(RuntimeError):
    pass


async def ingest_policy(
    policy_path: str | Path,
    embedding_provider: EmbeddingProvider,
    repository: ChunkRepository,
    *,
    embedding_model: str,
    embedding_dimensions: int,
) -> list[EmbeddedChunk]:
    markdown = Path(policy_path).read_text(encoding="utf-8")
    chunks = parse_policy(markdown)
    embeddings = await embedding_provider.embed([chunk.text for chunk in chunks])

    if len(embeddings) != len(chunks):
        raise IngestionError(
            f"expected {len(chunks)} embeddings, received {len(embeddings)}"
        )

    embedded_chunks: list[EmbeddedChunk] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        if len(embedding) != embedding_dimensions:
            raise IngestionError(
                f"chunk {chunk.chunk_id} has {len(embedding)} dimensions; "
                f"expected {embedding_dimensions}"
            )

        embedded_chunks.append(
            EmbeddedChunk(
                **chunk.model_dump(),
                embedding=embedding,
                embedding_model=embedding_model,
            )
        )

    await repository.replace_document(embedded_chunks)
    return embedded_chunks
