from collections.abc import Sequence

from ollama import AsyncClient
from pydantic import ValidationError

from mini_rag_lab.domain.models import GenerationDecision, RetrievedChunk
from mini_rag_lab.prompts import SYSTEM_PROMPT
from mini_rag_lab.services.generation import (
    apply_generation_guardrails,
    currency_comparisons,
)


class GenerationError(RuntimeError):
    pass


class OllamaEmbeddingProvider:
    def __init__(self, host: str, model: str) -> None:
        self._client = AsyncClient(host=host)
        self._model = model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embed(
            model=self._model,
            input=list(texts),
        )
        return [list(embedding) for embedding in response.embeddings]


class OllamaAnswerGenerator:
    def __init__(self, host: str, model: str, *, thinking: bool = False) -> None:
        self._client = AsyncClient(host=host)
        self._model = model
        self._thinking = thinking

    async def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> GenerationDecision:
        excerpts = "\n\n".join(
            (
                f"RELEVANCE RANK: {rank}\n"
                f"CHUNK ID: {chunk.chunk_id}\n"
                f"SECTION: {chunk.section}. {chunk.section_title}\n"
                f"TEXT:\n{chunk.text}"
            )
            for rank, chunk in enumerate(chunks, start=1)
        )
        comparisons = currency_comparisons(question, chunks)
        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{question}\n\n"
                        f"NUMERIC COMPARISONS:\n{comparisons}\n\n"
                        f"POLICY EXCERPTS:\n{excerpts}"
                    ),
                },
            ],
            format=GenerationDecision.model_json_schema(),
            think=self._thinking,
            options={"temperature": 0, "seed": 42},
        )

        content = response.message.content
        if content is None:
            raise GenerationError("generation model returned no content")

        try:
            decision = GenerationDecision.model_validate_json(content)
        except ValidationError as error:
            raise GenerationError(
                "generation model returned an invalid decision"
            ) from error
        return apply_generation_guardrails(question, chunks, decision)
