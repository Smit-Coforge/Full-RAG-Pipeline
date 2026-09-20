from mini_rag_lab.domain.models import (
    REFUSAL_ANSWER,
    AskRequest,
    AskResponse,
    Citation,
    RetrievedChunk,
    RetrievedChunkSummary,
)
from mini_rag_lab.domain.ports import (
    AnswerGenerator,
    ChunkRepository,
    EmbeddingProvider,
)
from mini_rag_lab.services.retrieval import retrieve_chunks

GENERATION_CONTEXT_SIZE = 1


class GroundedQueryService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: ChunkRepository,
        answer_generator: AnswerGenerator,
        *,
        embedding_dimensions: int,
        max_cosine_distance: float,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._repository = repository
        self._answer_generator = answer_generator
        self._embedding_dimensions = embedding_dimensions
        self._max_cosine_distance = max_cosine_distance

    async def ask(self, question: str) -> AskResponse:
        request = AskRequest(question=question)
        chunks = await retrieve_chunks(
            request.question,
            self._embedding_provider,
            self._repository,
            embedding_dimensions=self._embedding_dimensions,
        )
        summaries = [
            RetrievedChunkSummary(
                section=_section_label(chunk),
                distance=chunk.distance,
            )
            for chunk in chunks
        ]

        if not chunks or chunks[0].distance > self._max_cosine_distance:
            return _refusal_response(summaries)

        generation_context = chunks[:GENERATION_CONTEXT_SIZE]
        decision = await self._answer_generator.generate(
            request.question,
            generation_context,
        )
        supporting_chunk = next(
            (
                chunk
                for chunk in generation_context
                if chunk.chunk_id == decision.supporting_chunk_id
            ),
            None,
        )
        if supporting_chunk is None or decision.answer == REFUSAL_ANSWER:
            return _refusal_response(summaries)

        return AskResponse(
            answer=decision.answer,
            citation=Citation(
                document=supporting_chunk.document,
                version=supporting_chunk.version,
                section=_section_label(supporting_chunk),
            ),
            retrieved_chunks=summaries,
        )


def _section_label(chunk: RetrievedChunk) -> str:
    return f"{chunk.section}. {chunk.section_title}"


def _refusal_response(
    summaries: list[RetrievedChunkSummary],
) -> AskResponse:
    return AskResponse(
        answer=REFUSAL_ANSWER,
        citation=None,
        retrieved_chunks=summaries,
    )
