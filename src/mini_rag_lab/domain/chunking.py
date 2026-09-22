import re
from pathlib import Path

from mini_rag_lab.domain.extract import extract_document_text
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


# Top-level headings are "3. Title". A subsection such as "3.1 Detail" has no
# space after the first dot, so it stays in the parent section. The version
# line is matched on the word Version, not on a dash character.
_NUMBERED_HEADER_PATTERN = re.compile(
    r"^(?P<title>.+?)\s+Version\s+(?P<version>\d+(?:\.\d+)*)\s*$",
    re.IGNORECASE,
)
_NUMBERED_SECTION_PATTERN = re.compile(r"^(?P<section>\d+)\.\s+(?P<section_title>.+)$")
_TRAILING_SEPARATOR = re.compile(r"[^\w&)]+$", re.UNICODE)
_SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def _document_title(raw_title: str) -> str:
    return _TRAILING_SEPARATOR.sub("", raw_title.strip()).strip()


def _document_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
    if not slug:
        raise PolicyFormatError("document title does not produce a slug")
    return slug


def parse_numbered_sections(text: str) -> list[PolicyChunk]:
    """Split policy text on top-level numbered headings.

    Subsections such as 3.1 stay inside the parent section. This is the
    replaceable chunking step.
    """
    if not text.strip():
        raise PolicyFormatError("policy document is empty")

    lines = text.splitlines()
    header_index: int | None = None
    document = ""
    version = ""
    for index, line in enumerate(lines):
        header_match = _NUMBERED_HEADER_PATTERN.fullmatch(line.strip())
        if header_match is None:
            continue
        document = _document_title(header_match.group("title"))
        version = header_match.group("version")
        header_index = index
        break

    if header_index is None or not document:
        raise PolicyFormatError("policy title must include a document name and version")

    slug = _document_slug(document)
    chunks: list[PolicyChunk] = []
    current_section: str | None = None
    current_title: str | None = None
    body_lines: list[str] = []

    def append_current_section() -> None:
        if current_section is None or current_title is None:
            return

        body = "\n".join(body_lines).strip()
        if not body:
            raise PolicyFormatError(f"section {current_section} has no text")

        chunks.append(
            PolicyChunk(
                chunk_id=f"{slug}:v{version}:section-{current_section}",
                document=document,
                version=version,
                section=current_section,
                section_title=current_title,
                text=body,
            )
        )

    for line in lines[header_index + 1 :]:
        section_match = _NUMBERED_SECTION_PATTERN.fullmatch(line.strip())
        if section_match is not None:
            append_current_section()
            current_section = section_match.group("section")
            current_title = section_match.group("section_title").strip()
            body_lines = []
        elif current_section is not None:
            body_lines.append(line)
        elif line.strip():
            raise PolicyFormatError("unexpected text before the first policy section")

    append_current_section()
    if not chunks:
        raise PolicyFormatError("policy document has no numbered sections")
    return chunks


def load_policy_file(path: Path) -> list[PolicyChunk]:
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise PolicyFormatError(
            f"unsupported policy file type: {suffix or document_path.name}"
        )
    return parse_numbered_sections(extract_document_text(document_path))
