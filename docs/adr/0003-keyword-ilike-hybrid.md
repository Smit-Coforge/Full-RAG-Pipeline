# ADR 0003: Keyword search with Postgres ILIKE

- Date: 2026-09-23

## Decision

Hybrid retrieval keeps pgvector cosine search for meaning and adds a second
branch that matches the question against `policy_chunks.text` and
`section_title` with SQL `ILIKE`. The two lists are unioned by `chunk_id`.
Keyword hits are listed first so an exact match stays inside the final top-k.
There is no Chroma collection and no BM25 index.

The keyword branch extracts section codes such as `7.1` and content words of
four or more letters, drops sklearn English stopwords plus a small extra set
(`policy`, `section`, and auxiliaries like `does` that sklearn omits), then
OR-matches each remaining term with an escaped `ILIKE` pattern. When the
question includes a section code, rows that contain that code are ordered ahead
of rows that only matched a content word.

## Why

The graded hybrid case is a question that names an exact code or rare phrase.
Vector search can miss that chunk. A literal substring match pulls it back in.
`ILIKE` runs on the same rows ingest already wrote, so the keyword path cannot
drift from the vector store.

BM25 was rejected for this lab: tokenizers often split identifiers like `7.1`,
it needs a separate in-app index, and the corpus is only tens of chunks. Postgres
full-text (`tsvector`) has the same identifier weakness for the demo query.
Reranking can order candidates later; keyword search only has to recover the
missed chunk.

## What stays

- Vector search remains `ORDER BY embedding <=> query`.
- Candidate pool size is 8 per branch; ask keeps the merged top 3 until a
  reranker is added.
- Ask defaults to `hybrid`. Tests can pin `vector` or `keyword` when comparing
  strategies.
