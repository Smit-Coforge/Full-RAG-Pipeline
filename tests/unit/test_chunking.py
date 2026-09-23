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


def _policy(section_body: str) -> str:
    return f"HR Policy — Version 1.0\n\n1. Rules\n{section_body}\n"


def _words(count: int, *, sentence: bool = False) -> str:
    words = ["alpha"] * count
    if sentence:
        words[-1] = "alpha."
    return " ".join(words)


def test_section_at_token_cap_stays_one_chunk() -> None:
    chunks = parse_numbered_sections(_policy(_words(394)))

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "hr-policy:v1.0:section-1"
    assert len(chunks[0].text.split()) == 394


def test_section_over_token_cap_overlaps_the_previous_window() -> None:
    chunks = parse_numbered_sections(_policy(_words(395)))

    assert [chunk.chunk_id for chunk in chunks] == [
        "hr-policy:v1.0:section-1",
        "hr-policy:v1.0:section-1-part-2",
    ]
    assert {chunk.section for chunk in chunks} == {"1"}
    assert {chunk.section_title for chunk in chunks} == {"Rules"}
    first, second = (chunk.text.split() for chunk in chunks)
    assert first == ["alpha"] * 394
    assert second[:61] == first[-61:]
    assert second[-1] == "alpha"
    assert round(len(second) * 1.3) <= 512


def test_long_section_cuts_at_the_nearest_sentence() -> None:
    body = " ".join(_words(100, sentence=True) for _ in range(5))
    chunks = parse_numbered_sections(_policy(body))
    body_words = body.split()
    first, second = (chunk.text.split() for chunk in chunks)

    assert len(chunks) == 2
    assert first == body_words[:300]
    assert first[-1] == "alpha."
    assert second == body_words[239:]
    assert second[:61] == first[-61:]


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
