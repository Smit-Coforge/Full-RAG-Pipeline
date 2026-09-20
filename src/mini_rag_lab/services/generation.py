import re
from collections.abc import Sequence
from decimal import Decimal

from mini_rag_lab.domain.models import GenerationDecision, RetrievedChunk

_CURRENCY_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")


def currency_comparisons(
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> str:
    question_amounts = _CURRENCY_PATTERN.findall(question)
    policy_amounts = _CURRENCY_PATTERN.findall(
        "\n".join(chunk.text for chunk in chunks)
    )
    if not question_amounts or not policy_amounts:
        return "None."

    comparisons: list[str] = []
    for question_amount in question_amounts:
        for policy_amount in policy_amounts:
            left = Decimal(question_amount)
            right = Decimal(policy_amount)
            if left < right:
                relation = "less than"
            elif left > right:
                relation = "greater than"
            else:
                relation = "equal to"
            comparisons.append(f"${question_amount} is {relation} ${policy_amount}.")
    return "\n".join(comparisons)


def apply_numeric_threshold_guardrail(
    question: str,
    chunks: Sequence[RetrievedChunk],
    decision: GenerationDecision,
) -> GenerationDecision:
    question_amounts = _CURRENCY_PATTERN.findall(question)
    if not question_amounts:
        return decision

    for chunk in chunks:
        policy_text = chunk.text.lower()
        if "required" not in policy_text or "or more" not in policy_text:
            continue

        thresholds = _CURRENCY_PATTERN.findall(chunk.text)
        for question_amount in question_amounts:
            for threshold in thresholds:
                if Decimal(question_amount) < Decimal(threshold):
                    return GenerationDecision(
                        answer=(
                            f"No. ${question_amount} is below the ${threshold} "
                            "threshold, so the stated requirement does not apply "
                            "under this policy."
                        ),
                        supporting_chunk_id=chunk.chunk_id,
                    )

    return decision


def apply_policy_completeness_guardrail(
    question: str,
    chunks: Sequence[RetrievedChunk],
    decision: GenerationDecision,
) -> GenerationDecision:
    question_amounts = [
        Decimal(amount) for amount in _CURRENCY_PATTERN.findall(question)
    ]

    for chunk in chunks:
        policy_text = chunk.text.lower()
        has_default_and_exception = (
            "must" in policy_text
            and "requires" in policy_text
            and "approval" in policy_text
        )
        policy_amounts = [
            Decimal(amount) for amount in _CURRENCY_PATTERN.findall(chunk.text)
        ]
        exceeds_cap_requiring_approval = (
            "approve" in policy_text
            and "before booking" in policy_text
            and any(
                question_amount > policy_amount
                for question_amount in question_amounts
                for policy_amount in policy_amounts
            )
        )

        if has_default_and_exception or exceeds_cap_requiring_approval:
            return GenerationDecision(
                answer=" ".join(chunk.text.splitlines()),
                supporting_chunk_id=chunk.chunk_id,
            )

    return decision


def apply_generation_guardrails(
    question: str,
    chunks: Sequence[RetrievedChunk],
    decision: GenerationDecision,
) -> GenerationDecision:
    decision = apply_numeric_threshold_guardrail(question, chunks, decision)
    return apply_policy_completeness_guardrail(question, chunks, decision)
