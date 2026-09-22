import pytest

from mini_rag_lab.domain.chunking import PolicyFormatError, parse_numbered_sections

POLICY_TEXT = """\
HR Policy — Version 1.0

1. Purpose
This policy sets the rules.

2. Scope
The policy applies to all employees.
2.1 Contractors
Contractors follow the same rules.
"""


def test_subsection_stays_inside_its_parent_section() -> None:
    chunks = parse_numbered_sections(POLICY_TEXT)

    assert len(chunks) == 2
    assert [chunk.section for chunk in chunks] == ["1", "2"]
    assert chunks[0].model_dump() == {
        "chunk_id": "hr-policy:v1.0:section-1",
        "document": "HR Policy",
        "version": "1.0",
        "section": "1",
        "section_title": "Purpose",
        "text": "This policy sets the rules.",
    }
    assert chunks[1].model_dump() == {
        "chunk_id": "hr-policy:v1.0:section-2",
        "document": "HR Policy",
        "version": "1.0",
        "section": "2",
        "section_title": "Scope",
        "text": (
            "The policy applies to all employees.\n"
            "2.1 Contractors\n"
            "Contractors follow the same rules."
        ),
    }
    assert "Contractors follow the same rules." in chunks[1].text


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("HR Policy\n\n1. Purpose\nThis policy sets the rules.\n", "version"),
    ],
)
def test_invalid_policy_structure_is_rejected(text: str, message: str) -> None:
    with pytest.raises(PolicyFormatError, match=message):
        parse_numbered_sections(text)
