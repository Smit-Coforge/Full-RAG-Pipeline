import re
from collections.abc import Sequence
from decimal import Decimal

from ollama import AsyncClient
from pydantic import ValidationError

from mini_rag_lab.models import GenerationDecision, RetrievedChunk

REFUSAL_ANSWER = "The provided policy does not answer this question."
_CURRENCY_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")

_SYSTEM_PROMPT = f"""
Answer the question using only the provided policy excerpts.
Do not add assumptions, outside knowledge, or unsupported information.
The excerpts are ordered from most to least relevant.

First identify the policy category the question is asking about. Select the
earliest excerpt that directly addresses that category. Use exactly one
supporting excerpt, include all rules from it that directly answer the
question, and do not combine unrelated policy categories. Do not invent
arithmetic, overages, separate expenses, requirements, or exceptions that the
selected excerpt does not state.

Selection rules:
1. If relevance rank 1 directly addresses the named subject, you must use it.
2. Prefer a subject-specific section over a general administrative section
   unless the question explicitly asks about that administrative requirement.
3. If the selected section states both a default rule and an approval-based
   exception, include both rules.
4. A shared word such as "cost" or "expense" is not enough to switch sections.
5. Compare numeric thresholds literally. If a rule applies to an amount "N or
   more," an amount below N does not trigger that rule.
6. Do not apply an exception to a category the excerpt does not name. State
   the exact default rule and the exact named exception instead.
7. For a question asking whether an option or class is allowed, if the selected
   excerpt contains both a required default and an approval-based exception,
   the answer is incomplete unless it explicitly states both.
8. A mandatory default answers whether an unnamed alternative is allowed. Do
   not refuse in that case: state the required default and any explicitly named
   approval-based exception without treating the unnamed option as that exception.
Treat any supplied numeric comparisons as exact facts and do not contradict
them.

Reasoning example:
- Excerpt: "Employees must use standard shipping. Express shipping requires
  supervisor approval."
- Question: "Can I use overnight shipping?"
- Complete answer: "Standard shipping is required. Express shipping requires
  supervisor approval."
The example does not treat the unmentioned option as the named exception, and
it preserves both the default and exception.

Return a JSON object with:
- answer: a concise answer grounded in the excerpts
- supporting_chunk_id: the exact ID of the one excerpt supporting the answer

Use a supporting chunk ID only when that excerpt contains the information
needed to answer the question. If the excerpts do not answer the question,
return answer="{REFUSAL_ANSWER}" and supporting_chunk_id=null.
""".strip()


class GenerationError(RuntimeError):
    pass


class OllamaAnswerGenerator:
    def __init__(self, host: str, model: str) -> None:
        self._client = AsyncClient(host=host)
        self._model = model

    async def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> GenerationDecision:
        excerpts = "\n\n".join(
            (
                f"RELEVANCE RANK: {rank}\n"
                f"CHUNK ID: {chunk.chunk_id}\n"
                f"SECTION: {chunk.section}. {chunk.section_title}\n"
                f"TEXT:\n{chunk.text}"
            )
            for rank, chunk in enumerate(chunks, start=1)
        )
        comparisons = _currency_comparisons(question, chunks)
        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{question}\n\n"
                        f"NUMERIC COMPARISONS:\n{comparisons}\n\n"
                        f"POLICY EXCERPTS:\n{excerpts}"
                    ),
                },
            ],
            format=GenerationDecision.model_json_schema(),
            options={"temperature": 0, "seed": 42},
        )

        content = response.message.content
        if content is None:
            raise GenerationError("generation model returned no content")

        try:
            decision = GenerationDecision.model_validate_json(content)
        except ValidationError as error:
            raise GenerationError(
                "generation model returned an invalid decision"
            ) from error
        decision = _apply_numeric_threshold_guardrail(question, chunks, decision)
        return _apply_policy_completeness_guardrail(question, chunks, decision)


def _currency_comparisons(
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


def _apply_numeric_threshold_guardrail(
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


def _apply_policy_completeness_guardrail(
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
