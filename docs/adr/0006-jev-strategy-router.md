# ADR 0006: Jev closed strategy router (opt-in)

- Date: 2026-09-24

## Decision

Ask always retrieves with **hybrid** (vector + keyword → RRF → MiniLM). The
CLI has no `--strategy` flag.

Jev is an optional extra: `ask --jev` calls TypeSafe System One (`jev-latest`)
with one **Choice** question and may override the path to `vector`,
`keyword`, or `hybrid`. Criteria live in `adapters/jev.py`. Auth is
`TYPESAFE_API_KEY` against `https://api.typesafe.ai/v1/systemone`. Jev does
not rewrite the question, search, or generate the answer.

## Why

Hybrid is the reliable default for this corpus. An always-on router added
latency, cost, and wrong-path failures more often than it helped. Keeping Jev
behind `--jev` preserves the closed-router demo without hurting normal asks.

## What stays

- Default ask: hybrid only, no TypeSafe call.
- Missing `TYPESAFE_API_KEY` leaves the router unwired; `--jev` fails with a
  clear error instead of silently falling back.
- Keyword remains Postgres `ILIKE`; vector remains pgvector cosine; hybrid
  remains RRF + MiniLM. Unit tests may still call `retrieve_chunks` with an
  explicit strategy.
