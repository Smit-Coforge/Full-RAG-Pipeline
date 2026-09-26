from mini_rag_lab.adapters.cross_encoder import CrossEncoderReranker
from mini_rag_lab.domain.models import RetrievedChunk


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="Policy",
        version="1.0",
        section="1",
        section_title="Test",
        text=text,
        distance=0.5,
    )


def test_cross_encoder_reranks_relevant_passage_first() -> None:
    reranker = CrossEncoderReranker()
    question = "What is the shared refrigerator rule?"
    chunks = [
        _chunk("noise", "Hotel bookings require pre-approval before travel."),
        _chunk(
            "fridge",
            "No food in the shared refrigerator belongs to any employee.",
        ),
        _chunk("other", "Email tone must remain professional at all times."),
    ]

    ranked = reranker.rerank(question, chunks, limit=2)

    assert ranked[0].chunk_id == "fridge"
    assert ranked[0].rerank_score is not None
    assert ranked[1].rerank_score is not None
    assert ranked[0].rerank_score >= ranked[1].rerank_score
