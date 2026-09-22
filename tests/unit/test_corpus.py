from pathlib import Path

from mini_rag_lab.domain.chunking import load_policy_file

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"


def _only(paths: list[Path]) -> Path:
    assert len(paths) == 1
    return paths[0]


def test_hr_policy_versions_differ_and_health_policy_is_v1_only() -> None:
    assert CORPUS_DIR.is_dir()
    hr_v1 = load_policy_file(_only(sorted(CORPUS_DIR.glob("*HR Policy v1*"))))
    hr_v2 = load_policy_file(_only(sorted(CORPUS_DIR.glob("*HR Policy v2*"))))
    health_paths = sorted(
        path for path in CORPUS_DIR.iterdir() if "health" in path.name.casefold()
    )
    assert len(health_paths) == 1
    health = load_policy_file(health_paths[0])

    assert hr_v1 and hr_v2 and health
    assert {chunk.document for chunk in hr_v1} == {"HR Policy"}
    assert {chunk.document for chunk in hr_v2} == {"HR Policy"}
    assert {chunk.version for chunk in hr_v1} == {"1.0"}
    assert {chunk.version for chunk in hr_v2} == {"2.0"}
    assert {chunk.chunk_id for chunk in hr_v1}.isdisjoint(
        {chunk.chunk_id for chunk in hr_v2}
    )
    assert all(
        chunk.chunk_id == f"hr-policy:v{chunk.version}:section-{chunk.section}"
        for chunk in (*hr_v1, *hr_v2)
    )
    assert {chunk.document for chunk in health} == {"Health & Wellness Policy"}
    assert {chunk.version for chunk in health} == {"1.0"}
