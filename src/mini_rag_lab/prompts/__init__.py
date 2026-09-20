"""Grounded generation prompts loaded from text files."""

from pathlib import Path

from mini_rag_lab.domain.models import REFUSAL_ANSWER

SYSTEM_PROMPT = (
    Path(__file__)
    .with_name("system.txt")
    .read_text(encoding="utf-8")
    .replace("{refusal_answer}", REFUSAL_ANSWER)
    .strip()
)
