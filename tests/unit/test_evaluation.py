import asyncio

from mini_rag_lab.domain.models import (
    REFUSAL_ANSWER,
    AskResponse,
    Citation,
    RetrievedChunkSummary,
)
from mini_rag_lab.services.evaluation import (
    REQUIRED_CASES,
    EvaluationCase,
    evaluate_required_questions,
    score_case,
)


class FakeService:
    async def ask(self, question: str, *, use_jev: bool = False) -> AskResponse:
        case = next(case for case in REQUIRED_CASES if case.question == question)
        if case.is_refusal:
            return AskResponse(
                answer=REFUSAL_ANSWER,
                citation=None,
                retrieved_chunks=[
                    RetrievedChunkSummary(
                        document="HR Policy",
                        version="2.0",
                        section="1. Purpose",
                        distance=0.42,
                    )
                ],
            )

        assert case.expected_document is not None
        assert case.expected_version is not None
        assert case.expected_section is not None
        answer = " ".join(case.required_terms)
        return AskResponse(
            answer=answer,
            citation=Citation(
                document=case.expected_document,
                version=case.expected_version,
                section=case.expected_section,
            ),
            retrieved_chunks=[
                RetrievedChunkSummary(
                    document=case.expected_document,
                    version=case.expected_version,
                    section=case.expected_section,
                    distance=0.1,
                )
            ],
        )


def test_required_cases_cover_at_least_eight_questions() -> None:
    assert len(REQUIRED_CASES) >= 8
    assert sum(1 for case in REQUIRED_CASES if case.is_refusal) >= 1
    assert sum(1 for case in REQUIRED_CASES if not case.is_refusal) >= 7


def test_score_case_checks_retrieval_citation_and_terms() -> None:
    case = EvaluationCase(
        question="q",
        expected_document="HR Policy",
        expected_version="2.0",
        expected_section="7. Shared Refrigerator Policy",
        required_terms=("no food", "individual"),
    )
    response = AskResponse(
        answer="No food belongs to any individual employee.",
        citation=Citation(
            document="HR Policy",
            version="2.0",
            section="7. Shared Refrigerator Policy",
        ),
        retrieved_chunks=[
            RetrievedChunkSummary(
                document="HR Policy",
                version="2.0",
                section="7. Shared Refrigerator Policy",
                distance=0.1,
            )
        ],
    )
    checks = score_case(case, response)
    assert checks == {
        "expected_section_retrieved": True,
        "expected_section_cited": True,
        "answer_contains_expected_terms": True,
        "citation_in_retrieved": True,
    }


def test_score_case_fails_wrong_version_even_if_section_matches() -> None:
    case = EvaluationCase(
        question="q",
        expected_document="Preparedness Policy",
        expected_version="2.0",
        expected_section="4. Nuclear Apocalypse Protocol — Updated",
        required_terms=("refrigerator",),
    )
    response = AskResponse(
        answer="Hide under the desk, not the refrigerator.",
        citation=Citation(
            document="Preparedness Policy",
            version="1.0",
            section="4. Nuclear Apocalypse Protocol",
        ),
        retrieved_chunks=[
            RetrievedChunkSummary(
                document="Preparedness Policy",
                version="1.0",
                section="4. Nuclear Apocalypse Protocol",
                distance=0.1,
            )
        ],
    )
    checks = score_case(case, response)
    assert checks["expected_section_retrieved"] is False
    assert checks["expected_section_cited"] is False
    assert checks["answer_contains_expected_terms"] is True
    assert checks["citation_in_retrieved"] is True


def test_evaluator_reports_all_corpus_cases() -> None:
    result = asyncio.run(evaluate_required_questions(FakeService()))

    assert result["passed"] is True
    assert result["passed_count"] == len(REQUIRED_CASES)
    assert result["total"] == len(REQUIRED_CASES)
    assert result["total"] >= 8
    assert result["metrics"]["answerable_total"] == sum(
        1 for case in REQUIRED_CASES if not case.is_refusal
    )
    assert (
        result["metrics"]["retrieval_recall_pass_count"]
        == result["metrics"]["answerable_total"]
    )
    assert (
        result["metrics"]["gold_at_rank_1_count"]
        == result["metrics"]["answerable_total"]
    )
    assert "latency_mean_ms" in result["metrics"]
    assert all(case["passed"] for case in result["results"])
    assert all("latency_ms" in case for case in result["results"])
    assert all(
        case["gold_rank"] == 1
        for case in result["results"]
        if case["expected_section"] is not None
    )
