import re

from mini_rag_lab.domain.models import PolicyChunk

EXPECTED_SECTION_COUNT = 6
CHUNK_ID_PREFIX = "expense-policy"

_TITLE_PATTERN = re.compile(r"# (?P<document>.+) — Version (?P<version>\d+(?:\.\d+)*)")
_SECTION_PATTERN = re.compile(r"## (?P<section>\d+)\. (?P<section_title>.+)")


class PolicyFormatError(ValueError):
    pass


def parse_policy(markdown: str) -> list[PolicyChunk]:
    lines = markdown.splitlines()
    if not lines:
        raise PolicyFormatError("policy document is empty")

    title_match = _TITLE_PATTERN.fullmatch(lines[0])
    if title_match is None:
        raise PolicyFormatError("policy title must include a document name and version")

    document = title_match.group("document")
    version = title_match.group("version")
    chunks: list[PolicyChunk] = []
    current_section: str | None = None
    current_title: str | None = None
    body_lines: list[str] = []

    def append_current_section() -> None:
        if current_section is None or current_title is None:
            return

        text = "\n".join(body_lines).strip()
        if not text:
            raise PolicyFormatError(f"section {current_section} has no text")

        chunks.append(
            PolicyChunk(
                chunk_id=(f"{CHUNK_ID_PREFIX}:v{version}:section-{current_section}"),
                document=document,
                version=version,
                section=current_section,
                section_title=current_title,
                text=text,
            )
        )

    for line in lines[1:]:
        section_match = _SECTION_PATTERN.fullmatch(line)
        if section_match is not None:
            append_current_section()
            current_section = section_match.group("section")
            current_title = section_match.group("section_title")
            body_lines = []
        elif current_section is not None:
            body_lines.append(line)
        elif line.strip():
            raise PolicyFormatError("unexpected text before the first policy section")

    append_current_section()

    if len(chunks) != EXPECTED_SECTION_COUNT:
        raise PolicyFormatError(
            f"expected {EXPECTED_SECTION_COUNT} sections, found {len(chunks)}"
        )

    expected_sections = [str(number) for number in range(1, 7)]
    actual_sections = [chunk.section for chunk in chunks]
    if actual_sections != expected_sections:
        raise PolicyFormatError("policy sections must be numbered consecutively 1-6")

    return chunks
