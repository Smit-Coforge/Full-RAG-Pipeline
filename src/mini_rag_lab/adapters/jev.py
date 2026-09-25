"""Jev (TypeSafe) strategy router via System One API."""

from typing import Any, cast

import httpx

from mini_rag_lab.domain.models import RetrievalStrategy

DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"

_STRATEGY_CRITERIA: dict[str, str] = {
    "vector": """
Semantic search only. Use when the question is about one policy topic in
everyday or paraphrased language, and success does not depend on matching a
rare exact string in the chunk text. Typical cases: "can I wear a suit?",
"how does the shared fridge work?", "what is the caffeine limit?". Do not
use vector when the ask names a dotted section code (7.1, 4.3), a rare proper
noun that must hit literally (Doofenshmirtz, Perry), or asks to compare two
document versions (policy 1 vs 2).
""".strip(),
    "keyword": """
Substring (ILIKE) search only. Use only when an exact token in the question
should appear character-for-character in a chunk: a dotted section code
(7.1, 4.3), a rare proper noun (Doofenshmirtz, Perry), or another unique
literal phrase that semantic search might miss. Do not use keyword for
meta-words about the ask that are unlikely to appear in the policy body
(difference, compare, versus, update, change, version) unless those words
themselves are the literal policy terms being sought.
""".strip(),
    "hybrid": """
Run vector and keyword together, then merge. Prefer hybrid when: (1) the
question compares policies or versions (HR 1.0 vs 2.0, what changed);
(2) it mixes an exact code or rare name with a semantic ask; (3) either
path alone might miss evidence; or (4) you are unsure. When in doubt,
choose hybrid rather than keyword.
""".strip(),
}

_STRATEGY_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": """
Choose exactly one retrieval strategy for this policy question: vector,
keyword, or hybrid. Prefer hybrid unless the question is a clear
exact-token keyword case or a simple single-topic paraphrase with no
codes, rare names, or version comparison (vector).
""".strip(),
    "criteria": _STRATEGY_CRITERIA,
}


class JevRouterError(RuntimeError):
    pass


class JevStrategyRouter:
    """Closed router: returns only vector | keyword | hybrid."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_JEV_MODEL,
        systemone_url: str = DEFAULT_SYSTEMONE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise JevRouterError(
                "TYPESAFE_API_KEY is empty; set it in .env to use --jev"
            )
        self._api_key = api_key
        self._model = model
        self._url = systemone_url
        self._timeout = timeout_seconds

    async def choose(self, question: str) -> RetrievalStrategy:
        payload = {
            "model": self._model,
            "state": {"question": question},
            "questions": {"retrieval_strategy": _STRATEGY_QUESTION},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url,
                    headers=headers,
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise JevRouterError(f"Jev request failed: {exc}") from exc

        if response.status_code >= 400:
            raise JevRouterError(
                f"Jev HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise JevRouterError("Jev returned non-JSON body") from exc

        answer = body.get("answers", {}).get("retrieval_strategy")
        if not isinstance(answer, dict):
            raise JevRouterError(
                f"Jev response missing retrieval_strategy answer: {body!r}"
            )

        choice = answer.get("choice")
        if choice not in ("vector", "keyword", "hybrid"):
            raise JevRouterError(
                f"Jev returned unknown strategy {choice!r}; expected "
                "vector, keyword, or hybrid"
            )
        return cast(RetrievalStrategy, choice)
