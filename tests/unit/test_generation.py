import asyncio
from types import SimpleNamespace

from mini_rag_lab.adapters.ollama import OllamaAnswerGenerator
from mini_rag_lab.domain.models import (
    REFUSAL_ANSWER,
    GenerationDecision,
    RetrievedChunk,
)
from mini_rag_lab.prompts import SYSTEM_PROMPT
from mini_rag_lab.services.generation import (
    apply_numeric_threshold_guardrail,
    apply_policy_completeness_guardrail,
    currency_comparisons,
)


def _chunk(text: str, section: str = "1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{section}",
        document="Policy",
        version="1.0",
        section=section,
        section_title="Test",
        text=text,
        distance=0.1,
    )


def test_system_prompt_is_loaded_from_file() -> None:
    assert REFUSAL_ANSWER in SYSTEM_PROMPT
    assert "supporting_chunk_id" in SYSTEM_PROMPT


def test_currency_comparisons_are_computed_deterministically() -> None:
    result = currency_comparisons(
        "The cost is $20.",
        [_chunk("The limit is $25.")],
    )

    assert result == "$20 is less than $25."


def test_generator_forwards_thinking_configuration() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.arguments = {}

        async def chat(self, **kwargs):
            self.arguments = kwargs
            return SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer":"Grounded","supporting_chunk_id":"chunk-1"}'
                )
            )

    client = FakeClient()
    generator = OllamaAnswerGenerator(
        "http://ollama.test",
        "qwen3:8b",
        thinking=True,
    )
    generator._client = client

    decision = asyncio.run(
        generator.generate(
            "Question?",
            [_chunk("Grounded evidence.")],
        )
    )

    assert decision.answer == "Grounded"
    assert client.arguments["model"] == "qwen3:8b"
    assert client.arguments["think"] is True


def test_below_required_threshold_corrects_model_decision() -> None:
    chunk = _chunk("Receipts are required for expenses of $25 or more.", "5")
    incorrect = GenerationDecision(
        answer="A receipt is required.",
        supporting_chunk_id=chunk.chunk_id,
    )

    corrected = apply_numeric_threshold_guardrail(
        "Do I need a receipt for $20?",
        [chunk],
        incorrect,
    )

    assert corrected.answer.startswith("No.")
    assert "$20" in corrected.answer
    assert "$25" in corrected.answer
    assert corrected.supporting_chunk_id == chunk.chunk_id


def test_default_and_approval_exception_preserve_complete_policy() -> None:
    chunk = _chunk(
        "Employees must purchase economy airfare.\n"
        "Business-class airfare requires written approval from a vice president.",
        "3",
    )
    incomplete = GenerationDecision(
        answer="Economy only.",
        supporting_chunk_id=chunk.chunk_id,
    )

    corrected = apply_policy_completeness_guardrail(
        "Can I book first class?",
        [chunk],
        incomplete,
    )

    assert corrected.answer == " ".join(chunk.text.splitlines())
    assert corrected.supporting_chunk_id == chunk.chunk_id


def test_exceeded_cap_preserves_prebooking_approval_rule() -> None:
    chunk = _chunk(
        "Hotels are reimbursable up to $225 per night.\n"
        "A manager must approve higher rates before booking.",
        "2",
    )
    incomplete = GenerationDecision(
        answer="Ask a manager.",
        supporting_chunk_id=chunk.chunk_id,
    )

    corrected = apply_policy_completeness_guardrail(
        "My hotel is $250.",
        [chunk],
        incomplete,
    )

    assert corrected.answer == " ".join(chunk.text.splitlines())
