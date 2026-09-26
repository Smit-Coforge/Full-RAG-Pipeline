# ADR 0006: Jev closed strategy router (opt-in)

- Date: 2026-09-24
- Updated: 2026-09-25

## Decision

Ask always retrieves with **hybrid** (vector + keyword → RRF → MiniLM). The
CLI has no `--strategy` flag.

Jev is an optional extra: `ask --jev` / `evaluate --jev` calls TypeSafe
System One (`jev-latest`) with one **Choice** question and may override the
path to `vector`, `keyword`, or `hybrid`. Criteria live in `adapters/jev.py`.
Auth is `TYPESAFE_API_KEY` against TypeSafe System One. Jev does not rewrite
the question, search, or generate the answer. Whatever path it picks still
runs MiniLM rerank and grounded generation.

## Why

**Initial stance (architecture / theory).** The closed router can choose a
single-path `vector` or `keyword` run instead of full hybrid. On paper that
looked risky for this lab: always-hybrid already merges both signals via RRF,
so an LLM pick can drop one useful path, add a remote call, and fail on
edge queries. That is why Jev was shipped **opt-in only**, not as the product
default.

**Eval follow-up (this corpus).** On the ≥8-question harness, hybrid vs
`--jev` both scored **9/9** (same recall / citation / answer-term checks).
Latency for `--jev` was competitive and often slightly better (lower total /
mean / p95 on paired runs), which fits the architecture: when Jev picks a
single path it skips one retrieval leg + RRF, while rerank + generation still
dominate wall time. So Jev is not “a bad idea” on this tiny Doofenshmirtz
corpus — it looks fine next to always-hybrid.

**Still not the default.** The assignment corpus is very small and not a
representative RAG workload (few docs, short sections, limited conflict
surface). Latency and quality here are not enough evidence to make a remote
router the standard path. Default remains local hybrid (no API key, no
extra hop). Jev stays optional for demos and experiments.

## What stays

- Default ask / evaluate: hybrid only, no TypeSafe call.
- Missing `TYPESAFE_API_KEY` leaves the router unwired; `--jev` fails with a
  clear error instead of silently falling back.
- Keyword remains Postgres `ILIKE`; vector remains pgvector cosine; hybrid
  remains RRF + MiniLM. Unit tests may still call `retrieve_chunks` with an
  explicit strategy.
