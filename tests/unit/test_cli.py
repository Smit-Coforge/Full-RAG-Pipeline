import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mini_rag_lab import cli
from mini_rag_lab.domain.models import (
    AskResponse,
    Citation,
    EmbeddedChunk,
    RetrievedChunkSummary,
)


def test_migrate_command_reports_applied_files(monkeypatch, capsys) -> None:
    apply = AsyncMock(return_value=["001_create_policy_chunks.sql"])
    monkeypatch.setattr(cli, "apply_migrations", apply)
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://test"),
    )

    assert cli.main(["migrate", "--directory", "db"]) == 0

    apply.assert_awaited_once_with(
        "postgresql://test",
        Path("db"),
    )
    assert json.loads(capsys.readouterr().out) == {
        "applied": ["001_create_policy_chunks.sql"]
    }


def test_ingest_command_uses_runtime_adapters(monkeypatch, capsys) -> None:
    runtime = SimpleNamespace(
        embedding_provider=object(),
        repository=object(),
        settings=SimpleNamespace(
            embedding_model="model",
            embedding_dimensions=768,
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr(cli, "create_runtime", AsyncMock(return_value=runtime))
    chunk = EmbeddedChunk(
        chunk_id="chunk-1",
        document="Policy",
        version="1.0",
        section="1",
        section_title="Test",
        text="Text",
        embedding=[0.0] * 768,
        embedding_model="model",
    )
    ingest = AsyncMock(return_value=[chunk])
    monkeypatch.setattr(cli, "ingest_policy", ingest)

    assert cli.main(["ingest", "--policy", "custom.md"]) == 0

    ingest.assert_awaited_once_with(
        Path("custom.md"),
        runtime.embedding_provider,
        runtime.repository,
        embedding_model="model",
        embedding_dimensions=768,
    )
    runtime.close.assert_awaited_once()
    assert json.loads(capsys.readouterr().out) == {
        "document": "Policy",
        "version": "1.0",
        "chunks_stored": 1,
    }


def test_evaluate_command_returns_failure_exit_code(monkeypatch, capsys) -> None:
    runtime = SimpleNamespace(service=object(), close=AsyncMock())
    monkeypatch.setattr(cli, "create_runtime", AsyncMock(return_value=runtime))
    evaluate = AsyncMock(
        return_value={
            "passed": False,
            "passed_count": 5,
            "total": 6,
            "results": [],
        }
    )
    monkeypatch.setattr(cli, "evaluate_required_questions", evaluate)

    assert cli.main(["evaluate"]) == 1

    evaluate.assert_awaited_once_with(runtime.service)
    runtime.close.assert_awaited_once()
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_ask_command_prints_structured_json(monkeypatch, capsys) -> None:
    response = AskResponse(
        answer="Grounded answer",
        citation=Citation(
            document="Policy",
            version="1.0",
            section="1. Test",
        ),
        retrieved_chunks=[RetrievedChunkSummary(section="1. Test", distance=0.1)],
    )
    service = SimpleNamespace(ask=AsyncMock(return_value=response))
    runtime = SimpleNamespace(service=service, close=AsyncMock())
    monkeypatch.setattr(cli, "create_runtime", AsyncMock(return_value=runtime))

    assert cli.main(["ask", "What is covered?"]) == 0

    service.ask.assert_awaited_once_with("What is covered?")
    runtime.close.assert_awaited_once()
    assert json.loads(capsys.readouterr().out) == response.model_dump(mode="json")
