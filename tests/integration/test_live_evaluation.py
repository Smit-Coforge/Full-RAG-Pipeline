"""Live corpus evaluation harness (Postgres + Ollama). Skipped unless opted in."""

import asyncio
import os

import pytest

from mini_rag_lab.runtime import create_runtime
from mini_rag_lab.services.evaluation import (
    REQUIRED_CASES,
    evaluate_required_questions,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1",
        reason="set RUN_LIVE_TESTS=1 to use PostgreSQL and Ollama",
    ),
]


async def _run_live_evaluate() -> None:
    runtime = await create_runtime()
    try:
        result = await evaluate_required_questions(runtime.service, use_jev=False)
    finally:
        await runtime.close()

    assert result["total"] == len(REQUIRED_CASES)
    assert result["total"] >= 8
    assert result["passed"] is True, result.get("metrics")


def test_corpus_evaluate_harness_passes() -> None:
    asyncio.run(_run_live_evaluate())
