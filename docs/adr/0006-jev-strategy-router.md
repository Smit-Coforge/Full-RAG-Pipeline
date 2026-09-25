# ADR 0006: Jev closed strategy router (opt-in)

- Date: 2026-09-24

## Decision

Jev (TypeSafe System One API, model `jev-latest`) may choose the retrieval
strategy for an ask. We use one **Choice** question (not Noul or Score). The
Choice returns exactly one of `vector`, `keyword`, or `hybrid`. Criteria live
in `adapters/jev.py` (`_STRATEGY_QUESTION`). Auth is `TYPESAFE_API_KEY`
against `https://api.typesafe.ai/v1/systemone`. Jev does not rewrite the
question, search, or generate the answer.

The CLI keeps `--strategy` unchanged. `--router` turns the Jev path on; when
it is off, `--strategy` selects the path as before. When `--router` is on,
`--strategy` is ignored.

Keyword is reserved for exact section codes / rare literals. Compare-version
and ambiguous asks should land on hybrid.

## Why

The lab wants an agentic *closed* router, not SQL or multi-turn search. Jev
returns a typed Choice with probabilities, so the app branches without parsing
prose. Pinning strategy in evals and demos stays available by omitting
`--router`. TypeSafe billing (including free credits) applies via the native
API key rather than OpenRouter.

## What stays

- Default ask path remains `--strategy hybrid` with no TypeSafe call.
- Missing `TYPESAFE_API_KEY` leaves the router unwired; `--router` fails
  with a clear error instead of silently falling back.
- Keyword remains Postgres `ILIKE`; vector remains pgvector cosine; hybrid
  remains RRF + MiniLM.
