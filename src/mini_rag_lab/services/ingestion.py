from collections.abc import Sequence
from pathlib import Path

from mini_rag_lab.domain.chunking import load_policy_file, parse_policy
from mini_rag_lab.domain.models import EmbeddedChunk, PolicyChunk
from mini_rag_lab.domain.ports import ChunkRepository, EmbeddingProvider


class IngestionError(RuntimeError):
    pass


_CORPUS_SUFFIXES = {".pdf", ".docx"}


async def _embed_chunks(
    chunks: Sequence[PolicyChunk],
    embedding_provider: EmbeddingProvider,
    *,
    embedding_model: str,
    embedding_dimensions: int,
) -> list[EmbeddedChunk]:
    embeddings = await embedding_provider.embed(
        [chunk.text for chunk in chunks],
        task="search_document",
    )

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

    return embedded_chunks


async def ingest_policy(
    policy_path: str | Path,
    embedding_provider: EmbeddingProvider,
    repository: ChunkRepository,
    *,
    embedding_model: str,
    embedding_dimensions: int,
) -> list[EmbeddedChunk]:
    """Optional Markdown path: exactly six ``##`` sections (legacy mini-lab).

    Prefer :func:`ingest_corpus` for the default PDF/DOCX product path.
    """
    markdown = Path(policy_path).read_text(encoding="utf-8")
    embedded_chunks = await _embed_chunks(
        parse_policy(markdown),
        embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    await repository.replace_document(embedded_chunks)
    return embedded_chunks


async def ingest_corpus(
    corpus_path: str | Path,
    embedding_provider: EmbeddingProvider,
    repository: ChunkRepository,
    *,
    embedding_model: str,
    embedding_dimensions: int,
) -> list[EmbeddedChunk]:
    root = Path(corpus_path)
    if not root.is_dir():
        raise IngestionError(f"corpus directory not found: {root}")

    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in _CORPUS_SUFFIXES
    )
    if not files:
        raise IngestionError(f"no PDF or DOCX files in {root}")

    stored: list[EmbeddedChunk] = []
    for path in files:
        embedded_chunks = await _embed_chunks(
            load_policy_file(path),
            embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )
        await repository.replace_document(embedded_chunks)
        stored.extend(embedded_chunks)
    return stored
