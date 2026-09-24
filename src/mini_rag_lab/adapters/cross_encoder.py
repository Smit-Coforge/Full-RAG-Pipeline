"""Local CrossEncoder reranker (sentence-transformers)."""

from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import CrossEncoder

from mini_rag_lab.domain.models import RetrievedChunk

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """Scores (question, chunk) pairs; swappable via the Reranker port."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> None:
        self._model_name = model_name

    def rerank(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if limit < 1:
            raise ValueError("rerank limit must be at least 1")

        model = _load_cross_encoder(self._model_name)
        pairs = [(question, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)  # type: ignore[arg-type]
        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            chunk.model_copy(update={"rerank_score": float(score)})
            for chunk, score in ranked[:limit]
        ]
