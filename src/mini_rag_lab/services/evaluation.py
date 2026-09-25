import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mini_rag_lab.config import Settings
from mini_rag_lab.domain.models import (
    REFUSAL_ANSWER,
    AskResponse,
    Citation,
    RetrievedChunkSummary,
)
from mini_rag_lab.services.query import GroundedQueryService

DEFAULT_EVAL_RUNS_DIR = Path("eval_runs")

METRIC_LABELS: dict[str, str] = {
    "retrieval_recall_pass_count": "Retrieval recall",
    "citation_pass_count": "Citation accuracy",
    "answer_terms_pass_count": "Answer accuracy",
    "citation_in_retrieved_pass_count": "Cited chunk in retrieved list",
    "gold_at_rank_1_count": "Gold section ranked #1",
    "mean_gold_rank": "Mean gold rank (lower better)",
    "answerable_total": "Answerable cases",
    "refusal_pass_count": "Refusal accuracy",
    "refusal_total": "Refusal cases",
    "latency_total_ms": "Total ask latency (ms)",
    "latency_mean_ms": "Mean ask latency (ms)",
    "latency_p50_ms": "p50 ask latency (ms)",
    "latency_p95_ms": "p95 ask latency (ms)",
}


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Fixed corpus case: gold attribution plus answer-term checks."""

    question: str
    expected_document: str | None
    expected_version: str | None
    expected_section: str | None
    required_terms: tuple[str, ...] = ()

    @property
    def is_refusal(self) -> bool:
        return self.expected_section is None


# ≥8 fixed Doofenshmirtz corpus cases (retrieve → cite → answer terms / refuse).
REQUIRED_CASES = (
    EvaluationCase(
        question="Who owns food left in the shared refrigerator?",
        expected_document="HR Policy",
        expected_version="2.0",
        expected_section="7. Shared Refrigerator Policy",
        required_terms=("no food", "individual"),
    ),
    EvaluationCase(
        question="Must internal emails include a joke?",
        expected_document="HR Policy",
        expected_version="2.0",
        expected_section="3. Email Tone Requirement",
        required_terms=("joke",),
    ),
    EvaluationCase(
        question="Can I wear a suit to the office?",
        expected_document="HR Policy",
        expected_version="2.0",
        expected_section="4. Dress Code",
        required_terms=("formal", "prohibited"),
    ),
    EvaluationCase(
        question="What is the daily caffeine limit?",
        expected_document="Health & Wellness Policy",
        expected_version="1.0",
        expected_section="5. Caffeine Guidelines",
        required_terms=("400",),
    ),
    EvaluationCase(
        question="How often must employees train at the gym each week?",
        expected_document="Health & Wellness Policy",
        expected_version="1.0",
        expected_section="3. Gym Routine Requirements",
        required_terms=("three", "45"),
    ),
    EvaluationCase(
        question="Where should employees shelter during a nuclear apocalypse?",
        expected_document="Preparedness Policy",
        expected_version="2.0",
        expected_section="4. Nuclear Apocalypse Protocol — Updated",
        required_terms=("refrigerator",),
    ),
    EvaluationCase(
        question="How many tokens does each employee get per cycle?",
        expected_document="Time & Usage Policy",
        expected_version="2.0",
        expected_section="6. Token Allocation",
        required_terms=("500,000",),
    ),
    EvaluationCase(
        question="What happens to tokens when I win a foosball match?",
        expected_document="Time & Usage Policy",
        expected_version="2.0",
        expected_section="4. Foosball Time and the Winner-Takes-Tokens Rule",
        required_terms=("winner", "token"),
    ),
    EvaluationCase(
        question="What is the company's 401(k) matching percentage?",
        expected_document=None,
        expected_version=None,
        expected_section=None,
    ),
)


def _matches_gold(
    summary: RetrievedChunkSummary,
    case: EvaluationCase,
) -> bool:
    return (
        summary.document == case.expected_document
        and summary.version == case.expected_version
        and summary.section == case.expected_section
    )


def _citation_matches(citation: Citation | None, case: EvaluationCase) -> bool:
    if citation is None:
        return False
    return (
        citation.document == case.expected_document
        and citation.version == case.expected_version
        and citation.section == case.expected_section
    )


def _citation_in_retrieved(
    citation: Citation | None,
    chunks: list[RetrievedChunkSummary],
) -> bool:
    if citation is None:
        return False
    return any(
        summary.document == citation.document
        and summary.version == citation.version
        and summary.section == citation.section
        for summary in chunks
    )


def _gold_rank(case: EvaluationCase, chunks: list[RetrievedChunkSummary]) -> int | None:
    for index, summary in enumerate(chunks, start=1):
        if _matches_gold(summary, case):
            return index
    return None


def score_case(case: EvaluationCase, response: AskResponse) -> dict[str, bool]:
    """Score one case for retrieval, citation, and answer terms."""
    chunks = response.retrieved_chunks
    if case.is_refusal:
        return {
            "exact_refusal": response.answer == REFUSAL_ANSWER,
            "no_citation": response.citation is None,
        }

    answer = response.answer.lower()
    return {
        "expected_section_retrieved": any(
            _matches_gold(summary, case) for summary in chunks
        ),
        "expected_section_cited": _citation_matches(response.citation, case),
        "answer_contains_expected_terms": all(
            term.lower() in answer for term in case.required_terms
        ),
        "citation_in_retrieved": _citation_in_retrieved(response.citation, chunks),
    }


def _round_ms(value: float) -> float:
    return round(value, 1)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


async def evaluate_required_questions(
    service: GroundedQueryService,
    *,
    use_jev: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in REQUIRED_CASES:
        started = time.perf_counter()
        response = await service.ask(case.question, use_jev=use_jev)
        latency_ms = _round_ms((time.perf_counter() - started) * 1000)
        checks = score_case(case, response)
        gold_rank = (
            None if case.is_refusal else _gold_rank(case, response.retrieved_chunks)
        )
        results.append(
            {
                "question": case.question,
                "expected_document": case.expected_document,
                "expected_version": case.expected_version,
                "expected_section": case.expected_section,
                "passed": all(checks.values()),
                "checks": checks,
                "gold_rank": gold_rank,
                "latency_ms": latency_ms,
                "retrieval_strategy": response.retrieval_strategy,
                "response": response.model_dump(mode="json"),
            }
        )

    passed_count = sum(result["passed"] for result in results)
    answerable = [
        result for result in results if result["expected_section"] is not None
    ]
    refusal = [result for result in results if result["expected_section"] is None]
    latencies = sorted(float(result["latency_ms"]) for result in results)
    gold_ranks = [
        int(result["gold_rank"])
        for result in answerable
        if result.get("gold_rank") is not None
    ]
    return {
        "passed": passed_count == len(results),
        "passed_count": passed_count,
        "total": len(results),
        "metrics": {
            "retrieval_recall_pass_count": sum(
                1
                for result in answerable
                if result["checks"].get("expected_section_retrieved")
            ),
            "citation_pass_count": sum(
                1
                for result in answerable
                if result["checks"].get("expected_section_cited")
            ),
            "answer_terms_pass_count": sum(
                1
                for result in answerable
                if result["checks"].get("answer_contains_expected_terms")
            ),
            "citation_in_retrieved_pass_count": sum(
                1
                for result in answerable
                if result["checks"].get("citation_in_retrieved")
            ),
            "gold_at_rank_1_count": sum(1 for rank in gold_ranks if rank == 1),
            "mean_gold_rank": (
                _round_ms(sum(gold_ranks) / len(gold_ranks)) if gold_ranks else None
            ),
            "answerable_total": len(answerable),
            "refusal_pass_count": sum(1 for result in refusal if result["passed"]),
            "refusal_total": len(refusal),
            "latency_total_ms": _round_ms(sum(latencies)),
            "latency_mean_ms": (
                _round_ms(sum(latencies) / len(latencies)) if latencies else 0.0
            ),
            "latency_p50_ms": _round_ms(_percentile(latencies, 50)),
            "latency_p95_ms": _round_ms(_percentile(latencies, 95)),
        },
        "results": results,
    }


def evaluation_run_metadata(
    settings: Settings,
    *,
    use_jev: bool = False,
) -> dict[str, Any]:
    """Record models used for a run. No URLs, API keys, or passwords."""
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "generation_model": settings.generation_model,
        "generation_thinking": settings.generation_thinking,
        "max_cosine_distance": settings.max_cosine_distance,
        "reranker_model": settings.reranker_model,
        "jev": settings.jev_model if use_jev else "not used",
    }
    return metadata


def _md_cell(value: object) -> str:
    text = " " if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _cite_label(
    document: str | None,
    version: str | None,
    section: str | None,
) -> str:
    if section is None:
        return "refusal"
    return f"{document} v{version} · {section}"


def _checks_summary(checks: dict[str, bool]) -> str:
    return ", ".join(f"{'✓' if ok else '✗'} {name}" for name, ok in checks.items())


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """Markdown table with metric key, human label, and value."""
    headers = ("Metric", "Meaning", "Value")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for key, value in metrics.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    _md_cell(key),
                    _md_cell(METRIC_LABELS.get(key, key)),
                    _md_cell(value),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def format_results_table(results: list[dict[str, Any]]) -> str:
    """Markdown table: pass/fail, strategy, cites, rank, latency (answers in JSON)."""
    headers = (
        "#",
        "Pass",
        "Strategy",
        "Question",
        "Expected",
        "Cited",
        "Rank",
        "Latency ms",
        "Checks",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for index, result in enumerate(results, start=1):
        response = result.get("response") or {}
        citation = response.get("citation")
        if citation:
            cited = _cite_label(
                citation.get("document"),
                citation.get("version"),
                citation.get("section"),
            )
        else:
            cited = "none"
        strategy = result.get("retrieval_strategy") or response.get(
            "retrieval_strategy", "—"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _md_cell(index),
                    _md_cell("PASS" if result.get("passed") else "FAIL"),
                    _md_cell(strategy),
                    _md_cell(result.get("question")),
                    _md_cell(
                        _cite_label(
                            result.get("expected_document"),
                            result.get("expected_version"),
                            result.get("expected_section"),
                        )
                    ),
                    _md_cell(cited),
                    _md_cell(result.get("gold_rank") or "—"),
                    _md_cell(result.get("latency_ms")),
                    _md_cell(_checks_summary(result.get("checks") or {})),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def format_cli_metrics(result: dict[str, Any]) -> str:
    """Plain CLI lines: human label + value (full tables live in eval_runs/)."""
    metrics = result.get("metrics") or {}
    return "\n".join(
        f"{METRIC_LABELS.get(key, key)}: {value}" for key, value in metrics.items()
    )


def format_eval_report(
    result: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Human-readable Markdown report for one evaluate run."""
    meta_rows = "\n".join(
        f"| {_md_cell(key)} | {_md_cell(value)} |" for key, value in metadata.items()
    )
    stamp = metadata.get("timestamp_utc", "")
    return "\n".join(
        (
            f"# Eval run {stamp}",
            "",
            (
                f"**Overall:** "
                f"{'PASS' if result.get('passed') else 'FAIL'} "
                f"({result.get('passed_count')}/{result.get('total')})"
            ),
            "",
            "## Environment",
            "",
            "| Setting | Value |",
            "| --- | --- |",
            meta_rows,
            "",
            "## Metrics",
            "",
            format_metrics_table(result.get("metrics") or {}),
            "",
            "## Results",
            "",
            format_results_table(result.get("results") or []),
            "",
            "## Full JSON",
            "",
            "```json",
            json.dumps(result, indent=2, ensure_ascii=False),
            "```",
            "",
        )
    )


def write_eval_run(
    result: dict[str, Any],
    settings: Settings,
    *,
    use_jev: bool = False,
    output_dir: Path = DEFAULT_EVAL_RUNS_DIR,
) -> Path:
    """Write a timestamped Markdown report under eval_runs/ (gitignored)."""
    metadata = evaluation_run_metadata(settings, use_jev=use_jev)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eval-{stamp}.md"
    path.write_text(format_eval_report(result, metadata), encoding="utf-8")
    return path
