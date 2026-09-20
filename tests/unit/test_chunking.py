from pathlib import Path

import pytest

from mini_rag_lab.domain.chunking import PolicyFormatError, parse_policy


def test_policy_is_split_into_six_structural_chunks() -> None:
    chunks = parse_policy(Path("policy.md").read_text(encoding="utf-8"))

    assert len(chunks) == 6
    assert [chunk.section for chunk in chunks] == ["1", "2", "3", "4", "5", "6"]
    assert chunks[0].model_dump() == {
        "chunk_id": "expense-policy:v2.0:section-1",
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1",
        "section_title": "Meals",
        "text": (
            "Employees may claim up to $65 per day for meals while traveling "
            "overnight.\nAlcohol is not reimbursable."
        ),
    }
    assert chunks[-1].section_title == "Submission Deadline"


@pytest.mark.parametrize(
    ("markdown", "message"),
    [
        ("", "empty"),
        ("# Policy\n", "title"),
        (
            (
                "# Employee Expense Policy — Version 2.0\n\n"
                "unexpected text\n## 1. Meals\nText"
            ),
            "unexpected text",
        ),
        (
            ("# Employee Expense Policy — Version 2.0\n\n## 1. Meals\nText"),
            "expected 6 sections",
        ),
    ],
)
def test_invalid_policy_structure_is_rejected(markdown: str, message: str) -> None:
    with pytest.raises(PolicyFormatError, match=message):
        parse_policy(markdown)
