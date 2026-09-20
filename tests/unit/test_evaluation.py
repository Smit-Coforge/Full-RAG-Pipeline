import asyncio

from mini_rag_lab.domain.models import (
    REFUSAL_ANSWER,
    AskResponse,
    Citation,
    RetrievedChunkSummary,
)
from mini_rag_lab.services.evaluation import REQUIRED_CASES, evaluate_required_questions


class FakeService:
    async def ask(self, question: str) -> AskResponse:
        case = next(case for case in REQUIRED_CASES if case.question == question)
        if case.expected_section is None:
            return AskResponse(
                answer=REFUSAL_ANSWER,
                citation=None,
                retrieved_chunks=[
                    RetrievedChunkSummary(
                        section="4. Ground Transportation",
                        distance=0.42,
                    )
                ],
            )

        answer = " ".join(case.required_terms)
        return AskResponse(
            answer=answer,
            citation=Citation(
                document="Employee Expense Policy",
                version="2.0",
                section=case.expected_section,
            ),
            retrieved_chunks=[
                RetrievedChunkSummary(
                    section=case.expected_section,
                    distance=0.1,
                )
            ],
        )


def test_evaluator_reports_all_six_cases() -> None:
    result = asyncio.run(evaluate_required_questions(FakeService()))

    assert result["passed"] is True
    assert result["passed_count"] == 6
    assert result["total"] == 6
    assert all(case["passed"] for case in result["results"])
