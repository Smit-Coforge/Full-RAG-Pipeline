# ADR 0005: RRF merge plus MiniLM CrossEncoder rerank

- Date: 2026-09-24

## Decision

After retrieval, candidates are fused with Reciprocal Rank Fusion (RRF, `k=60`)
when the strategy is hybrid. A local CrossEncoder then scores each
`(question, chunk.text)` pair and keeps the top 3. The locked model is
`cross-encoder/ms-marco-MiniLM-L-6-v2`, reached through a `Reranker` port so
BGE or a hosted API (for example Cohere) can replace it later without changing
hybrid or RRF.

Stored chunk text is unchanged. JSON summaries may include `rerank_score`.
Generation receives up to the final top 3 excerpts. When a rerank score is
present, the old cosine distance gate does not refuse the answer.

## Why

Keyword-first truncation let weak `ILIKE` hits occupy the final top 3 before
any semantic check. RRF keeps both lists' ranks without preferring keyword
order alone. A CrossEncoder is the assignment's practical reranker: it reads
the question and passage together, which bi-encoders and cosine distance do
not. MiniLM-L-6 is small enough for CPU Docker; quality comparisons with BGE
come later.

## What stays

- Vector search stays pgvector cosine with Nomic prefixes.
- Keyword search stays Postgres `ILIKE`.
- Cohere and other families are out of scope until a second adapter is added.
