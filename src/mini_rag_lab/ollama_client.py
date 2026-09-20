from collections.abc import Sequence

from ollama import AsyncClient


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
