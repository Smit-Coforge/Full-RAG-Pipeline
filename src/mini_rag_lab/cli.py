import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from mini_rag_lab.config import get_settings
from mini_rag_lab.migrations import apply_migrations
from mini_rag_lab.runtime import create_runtime
from mini_rag_lab.services.evaluation import evaluate_required_questions
from mini_rag_lab.services.ingestion import ingest_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-rag-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    migrate = commands.add_parser("migrate", help="apply SQL migrations")
    migrate.add_argument(
        "--directory",
        type=Path,
        default=Path("migrations"),
        help="directory containing SQL migrations",
    )

    ingest = commands.add_parser("ingest", help="embed and store policy sections")
    ingest.add_argument(
        "--policy",
        type=Path,
        default=Path("policy.md"),
        help="policy Markdown file",
    )

    ask = commands.add_parser("ask", help="ask one grounded policy question")
    ask.add_argument("question", help="question to answer")

    commands.add_parser("evaluate", help="run the six required questions")
    return parser


async def _migrate(args: argparse.Namespace) -> int:
    settings = get_settings()
    applied = await apply_migrations(settings.database_url, args.directory)
    print(json.dumps({"applied": applied}))
    return 0


async def _ingest(args: argparse.Namespace) -> int:
    runtime = await create_runtime()
    try:
        chunks = await ingest_policy(
            args.policy,
            runtime.embedding_provider,
            runtime.repository,
            embedding_model=runtime.settings.embedding_model,
            embedding_dimensions=runtime.settings.embedding_dimensions,
        )
    finally:
        await runtime.close()

    print(
        json.dumps(
            {
                "document": chunks[0].document,
                "version": chunks[0].version,
                "chunks_stored": len(chunks),
            }
        )
    )
    return 0


async def _evaluate() -> int:
    runtime = await create_runtime()
    try:
        result = await evaluate_required_questions(runtime.service)
    finally:
        await runtime.close()

    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


async def _ask(args: argparse.Namespace) -> int:
    runtime = await create_runtime()
    try:
        response = await runtime.service.ask(args.question)
    finally:
        await runtime.close()

    print(json.dumps(response.model_dump(mode="json"), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "migrate":
        return asyncio.run(_migrate(args))
    if args.command == "ingest":
        return asyncio.run(_ingest(args))
    if args.command == "evaluate":
        return asyncio.run(_evaluate())
    if args.command == "ask":
        return asyncio.run(_ask(args))
    raise AssertionError(f"unhandled command: {args.command}")


def run() -> None:
    raise SystemExit(main())
