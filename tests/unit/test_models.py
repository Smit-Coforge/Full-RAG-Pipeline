import pytest
from pydantic import ValidationError

from mini_rag_lab.config import Settings
from mini_rag_lab.domain.models import (
    AskRequest,
    AskResponse,
    EmbeddedChunk,
    RetrievedChunkSummary,
)


def test_request_is_trimmed_and_extra_fields_are_rejected() -> None:
    assert AskRequest(question="  What is covered?  ").question == "What is covered?"

    with pytest.raises(ValidationError):
        AskRequest(question="valid", unexpected=True)


def test_embedding_requires_exactly_768_values() -> None:
    with pytest.raises(ValidationError):
        EmbeddedChunk(
            chunk_id="chunk-1",
            document="Policy",
            version="1.0",
            section="1",
            section_title="Test",
            text="Text",
            embedding=[0.0] * 767,
            embedding_model="test",
        )


def test_response_rejects_more_than_three_retrieved_chunks() -> None:
    with pytest.raises(ValidationError):
        AskResponse(
            answer="Answer",
            citation=None,
            retrieved_chunks=[
                RetrievedChunkSummary(section=str(index), distance=float(index))
                for index in range(4)
            ],
        )


def test_settings_reject_invalid_fixed_dimensions_and_distance() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_dimensions=384)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_cosine_distance=3)
