import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mini_rag_lab.adapters.jev import JevRouterError, JevStrategyRouter
from mini_rag_lab.domain.models import (
    GenerationDecision,
    RetrievedChunk,
)
from mini_rag_lab.services.query import GroundedQueryService


def test_jev_router_returns_choice(monkeypatch) -> None:
    payload = {
        "answers": {
            "retrieval_strategy": {
                "type": "choice",
                "choice": "keyword",
                "confidence": 0.8,
                "probabilities": {"vector": 0.1, "keyword": 0.8, "hybrid": 0.1},
            }
        }
    }
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    response.text = ""

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: mock_client)

    router = JevStrategyRouter("sk-test")
    assert asyncio.run(router.choose("What does section 7.1 say?")) == "keyword"
    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.await_args
    assert call_kwargs.kwargs["json"]["questions"]["retrieval_strategy"]["type"] == (
        "choice"
    )


def test_jev_router_rejects_empty_key() -> None:
    with pytest.raises(JevRouterError, match="TYPESAFE_API_KEY"):
        JevStrategyRouter("   ")


def test_jev_router_rejects_unknown_choice(monkeypatch) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "answers": {"retrieval_strategy": {"type": "choice", "choice": "sql"}}
    }
    response.text = ""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: mock_client)

    router = JevStrategyRouter("sk-test")
    with pytest.raises(JevRouterError, match="unknown strategy"):
        asyncio.run(router.choose("hi"))


class _Embed:
    async def embed(self, texts, *, task="search_document"):
        return [[0.0] * 768 for _ in texts]


class _Repo:
    async def search(self, embedding, *, limit):
        return [
            RetrievedChunk(
                chunk_id="c1",
                document="Policy",
                version="1.0",
                section="1",
                section_title="A",
                text="text",
                distance=0.1,
            )
        ]

    async def keyword_search(self, question, *, limit):
        return []


class _Gen:
    async def generate(self, question, chunks):
        return GenerationDecision(answer="ok", supporting_chunk_id="c1")


class _Router:
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy
        self.calls = 0

    async def choose(self, question: str) -> str:
        self.calls += 1
        return self.strategy


def test_ask_use_router_overrides_strategy() -> None:
    router = _Router("vector")
    service = GroundedQueryService(
        _Embed(),
        _Repo(),
        _Gen(),
        embedding_dimensions=768,
        max_cosine_distance=0.4,
        strategy_router=router,
    )
    response = asyncio.run(
        service.ask("q", strategy="hybrid", use_router=True)
    )
    assert router.calls == 1
    assert response.retrieval_strategy == "vector"


def test_ask_without_router_keeps_strategy() -> None:
    router = _Router("keyword")
    service = GroundedQueryService(
        _Embed(),
        _Repo(),
        _Gen(),
        embedding_dimensions=768,
        max_cosine_distance=0.4,
        strategy_router=router,
    )
    response = asyncio.run(
        service.ask("q", strategy="vector", use_router=False)
    )
    assert router.calls == 0
    assert response.retrieval_strategy == "vector"


def test_ask_router_without_wiring_raises() -> None:
    service = GroundedQueryService(
        _Embed(),
        _Repo(),
        _Gen(),
        embedding_dimensions=768,
        max_cosine_distance=0.4,
    )
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        asyncio.run(service.ask("q", use_router=True))
