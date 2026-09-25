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


def test_ingest_command_defaults_to_corpus_directory(monkeypatch, capsys) -> None:
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
    chunks = [
        EmbeddedChunk(
            chunk_id=f"hr-policy:v{version}:section-1",
            document="HR Policy",
            version=version,
            section="1",
            section_title="Purpose",
            text=f"Body {version}",
            embedding=[0.0] * 768,
            embedding_model="model",
        )
        for version in ("1.0", "2.0")
    ]
    ingest_corpus = AsyncMock(return_value=chunks)
    ingest_policy = AsyncMock()
    monkeypatch.setattr(cli, "ingest_corpus", ingest_corpus)
    monkeypatch.setattr(cli, "ingest_policy", ingest_policy)

    assert cli.main(["ingest"]) == 0

    ingest_corpus.assert_awaited_once_with(
        Path("corpus"),
        runtime.embedding_provider,
        runtime.repository,
        embedding_model="model",
        embedding_dimensions=768,
    )
    ingest_policy.assert_not_awaited()
    runtime.close.assert_awaited_once()
    assert json.loads(capsys.readouterr().out) == {
        "documents": [
            {"document": "HR Policy", "version": "1.0", "chunks_stored": 1},
            {"document": "HR Policy", "version": "2.0", "chunks_stored": 1},
        ],
        "chunks_stored": 2,
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

    service.ask.assert_awaited_once_with(
        "What is covered?",
        use_jev=False,
    )
    runtime.close.assert_awaited_once()
    assert json.loads(capsys.readouterr().out) == response.model_dump(mode="json")


def test_ask_command_prints_unicode_without_escapes(monkeypatch, capsys) -> None:
    response = AskResponse(
        answer="Use the fridge",
        citation=Citation(
            document="Preparedness Policy",
            version="2.0",
            section="4. Nuclear Apocalypse Protocol — Updated",
        ),
        retrieved_chunks=[
            RetrievedChunkSummary(
                section="4. Nuclear Apocalypse Protocol — Updated",
                distance=0.2,
            )
        ],
    )
    service = SimpleNamespace(ask=AsyncMock(return_value=response))
    runtime = SimpleNamespace(service=service, close=AsyncMock())
    monkeypatch.setattr(cli, "create_runtime", AsyncMock(return_value=runtime))

    assert cli.main(["ask", "Where to shelter?"]) == 0

    raw = capsys.readouterr().out
    assert "\\u2014" not in raw
    assert "—" in raw
    assert json.loads(raw)["citation"]["section"].endswith("Updated")


def test_ask_command_passes_jev_flag(monkeypatch, capsys) -> None:
    response = AskResponse(
        answer="Grounded answer",
        citation=None,
        retrieved_chunks=[
            RetrievedChunkSummary(section="7. Refrigerator", distance=0.0)
        ],
        retrieval_strategy="keyword",
    )
    service = SimpleNamespace(ask=AsyncMock(return_value=response))
    runtime = SimpleNamespace(service=service, close=AsyncMock())
    monkeypatch.setattr(cli, "create_runtime", AsyncMock(return_value=runtime))

    assert (
        cli.main(
            [
                "ask",
                "What does section 7.1 say about the refrigerator?",
                "--jev",
            ]
        )
        == 0
    )

    service.ask.assert_awaited_once_with(
        "What does section 7.1 say about the refrigerator?",
        use_jev=True,
    )
