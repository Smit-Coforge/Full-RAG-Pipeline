# Corpus evaluation harness (Part 5)

Fixed ≥8 Doofenshmirtz corpus questions. Each answerable case checks:

1. **Retrieval recall** — gold `document` + `version` + `section` appears in
   the top retrieved summaries
2. **Citation** — the model cites that same gold triple
3. **Answer terms** — required substrings appear in the answer (case-insensitive)

One case expects the exact refusal string with `citation: null`.

## Cases

| # | Question (short) | Gold cite | Required terms |
| --- | --- | --- | --- |
| 1 | Shared refrigerator ownership | HR Policy v2.0 §7 | `no food`, `individual` |
| 2 | Email must include a joke | HR Policy v2.0 §3 | `joke` |
| 3 | Suit / formal attire | HR Policy v2.0 §4 | `formal`, `prohibited` |
| 4 | Daily caffeine limit | Health & Wellness v1.0 §5 | `400` |
| 5 | Weekly gym sessions | Health & Wellness v1.0 §3 | `three`, `45` |
| 6 | Nuclear shelter (prefer v2) | Preparedness v2.0 §4 | `refrigerator` |
| 7 | Tokens per cycle (prefer v2) | Time & Usage v2.0 §6 | `500,000` |
| 8 | Foosball winner-takes-tokens | Time & Usage v2.0 §4 | `winner`, `token` |
| 9 | 401(k) match (out of corpus) | refusal | exact refusal |

Cases 6–7 encode version preference: older conflicting sections must lose.

## Reproduce

```shell
docker compose up -d
python -m mini_rag_lab migrate
python -m mini_rag_lab ingest
python -m mini_rag_lab evaluate
```

Pytest covers scoring and a fake end-to-end pass of all cases
(`tests/unit/test_evaluation.py`). Live `evaluate` needs Ollama + a filled DB
and is not required in default CI.

## Scoring output shape

```json
{
  "passed": true,
  "passed_count": 9,
  "total": 9,
  "metrics": {
    "retrieval_recall_pass_count": 8,
    "citation_pass_count": 8,
    "answer_terms_pass_count": 8,
    "answerable_total": 8,
    "refusal_pass_count": 1,
    "refusal_total": 1
  },
  "results": []
}
```

Record a live run JSON under this file when you have a green `evaluate` against
the seeded corpus.

## Live run (25 September 2026)

`python -m mini_rag_lab evaluate` against a migrated + ingested `corpus/` —
**9/9 passed**.

| Question | Result | Citation |
| --- | --- | --- |
| Who owns food left in the shared refrigerator? | No food belongs to any individual | HR Policy v2.0 §7 |
| Must internal emails include a joke? | Must begin or end with a joke | HR Policy v2.0 §3 |
| Can I wear a suit to the office? | Formal attire prohibited | HR Policy v2.0 §4 |
| What is the daily caffeine limit? | 400 mg | Health & Wellness v1.0 §5 |
| How often must employees train at the gym each week? | Three sessions, 45 minutes | Health & Wellness v1.0 §3 |
| Where should employees shelter during a nuclear apocalypse? | Industrial refrigerator (v2) | Preparedness v2.0 §4 |
| How many tokens does each employee get per cycle? | 500,000 (v2) | Time & Usage v2.0 §6 |
| What happens to tokens when I win a foosball match? | Winner takes loser's tokens | Time & Usage v2.0 §4 |
| What is the company's 401(k) matching percentage? | Exact refusal | none |

```json
{
  "passed": true,
  "passed_count": 9,
  "total": 9,
  "metrics": {
    "retrieval_recall_pass_count": 8,
    "citation_pass_count": 8,
    "answer_terms_pass_count": 8,
    "answerable_total": 8,
    "refusal_pass_count": 1,
    "refusal_total": 1
  }
}
```
