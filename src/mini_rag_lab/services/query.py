from mini_rag_lab.domain.models import (
    REFUSAL_ANSWER,
    AskRequest,
    AskResponse,
    Citation,
    RetrievalStrategy,
    RetrievedChunk,
    RetrievedChunkSummary,
)
from mini_rag_lab.domain.ports import (
    AnswerGenerator,
    ChunkRepository,
    EmbeddingProvider,
    Reranker,
    StrategyRouter,
)
from mini_rag_lab.services.retrieval import retrieve_chunks

GENERATION_CONTEXT_SIZE = 3


class GroundedQueryService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: ChunkRepository,
        answer_generator: AnswerGenerator,
        *,
        embedding_dimensions: int,
        max_cosine_distance: float,
        reranker: Reranker | None = None,
        strategy_router: StrategyRouter | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._repository = repository
        self._answer_generator = answer_generator
        self._embedding_dimensions = embedding_dimensions
        self._max_cosine_distance = max_cosine_distance
        self._reranker = reranker
        self._strategy_router = strategy_router

    async def ask(
        self,
        question: str,
        *,
        strategy: RetrievalStrategy = "hybrid",
        use_router: bool = False,
    ) -> AskResponse:
        request = AskRequest(question=question)
        resolved = strategy
        if use_router:
            if self._strategy_router is None:
                raise RuntimeError(
                    "use_router=True but no strategy_router is configured; "
                    "set TYPESAFE_API_KEY in .env"
                )
            resolved = await self._strategy_router.choose(request.question)

        chunks = await retrieve_chunks(
            request.question,
            self._embedding_provider,
            self._repository,
            embedding_dimensions=self._embedding_dimensions,
            strategy=resolved,
            reranker=self._reranker,
        )
        summaries = [
            RetrievedChunkSummary(
                section=_section_label(chunk),
                distance=chunk.distance,
                rerank_score=chunk.rerank_score,
            )
            for chunk in chunks
        ]

        if not chunks:
            return _refusal_response(summaries, resolved)

        nearest = chunks[0]
        # After CrossEncoder, trust the rerank order. Cosine gate only applies
        # when the top chunk has no rerank_score (vector-only distance check).
        if (
            nearest.rerank_score is None
            and nearest.distance > self._max_cosine_distance
        ):
            return _refusal_response(summaries, resolved)

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
            return _refusal_response(summaries, resolved)

        return AskResponse(
            answer=decision.answer,
            citation=Citation(
                document=supporting_chunk.document,
                version=supporting_chunk.version,
                section=_section_label(supporting_chunk),
            ),
            retrieved_chunks=summaries,
            retrieval_strategy=resolved,
        )


def _section_label(chunk: RetrievedChunk) -> str:
    return f"{chunk.section}. {chunk.section_title}"


def _refusal_response(
    summaries: list[RetrievedChunkSummary],
    strategy: RetrievalStrategy,
) -> AskResponse:
    return AskResponse(
        answer=REFUSAL_ANSWER,
        citation=None,
        retrieved_chunks=summaries,
        retrieval_strategy=strategy,
    )
