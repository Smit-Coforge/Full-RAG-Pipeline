from dataclasses import dataclass
from typing import Any

from mini_rag_lab.domain.models import REFUSAL_ANSWER
from mini_rag_lab.services.query import GroundedQueryService


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    question: str
    expected_section: str | None
    required_terms: tuple[str, ...] = ()


REQUIRED_CASES = (
    EvaluationCase(
        "How much can I spend on food each day?",
        "1. Meals",
        ("$65",),
    ),
    EvaluationCase(
        "Can I book first-class airfare?",
        "3. Airfare",
        ("economy", "business", "approval"),
    ),
    EvaluationCase(
        "My hotel costs $250. What do I need?",
        "2. Hotels",
        ("manager", "approve", "before booking"),
    ),
    EvaluationCase(
        "Do I need a receipt for a $20 taxi?",
        "5. Receipts",
        ("no", "$25"),
    ),
    EvaluationCase(
        "Can I claim a limousine upgrade?",
        "4. Ground Transportation",
        ("not reimbursable",),
    ),
    EvaluationCase(
        "Does the company reimburse gym memberships?",
        None,
    ),
)


async def evaluate_required_questions(
    service: GroundedQueryService,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in REQUIRED_CASES:
        response = await service.ask(case.question)
        retrieved_sections = [chunk.section for chunk in response.retrieved_chunks]
        checks: dict[str, bool]
        if case.expected_section is None:
            checks = {
                "exact_refusal": response.answer == REFUSAL_ANSWER,
                "no_citation": response.citation is None,
            }
        else:
            answer = response.answer.lower()
            checks = {
                "expected_section_retrieved": (
                    case.expected_section in retrieved_sections
                ),
                "expected_section_cited": (
                    response.citation is not None
                    and response.citation.section == case.expected_section
                ),
                "answer_contains_expected_terms": all(
                    term.lower() in answer for term in case.required_terms
                ),
            }

        results.append(
            {
                "question": case.question,
                "passed": all(checks.values()),
                "checks": checks,
                "response": response.model_dump(mode="json"),
            }
        )

    passed_count = sum(result["passed"] for result in results)
    return {
        "passed": passed_count == len(results),
        "passed_count": passed_count,
        "total": len(results),
        "results": results,
    }
