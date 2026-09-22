"""Plain-text extraction for corpus PDF and DOCX files."""

from pathlib import Path

from docx import Document
from pypdf import PdfReader


def extract_document_text(path: Path) -> str:
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(document_path)
        pages = (page.extract_text() or "" for page in reader.pages)
        return "\n".join(pages)
    if suffix == ".docx":
        document = Document(str(document_path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError(f"unsupported policy file type: {suffix or document_path.name}")
