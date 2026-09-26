# ADR 0004: Nomic search_document / search_query prefixes

- Date: 2026-09-24

## Decision

When embedding with `nomic-embed-text`, the Ollama adapter prefixes each string:

- ingest uses `search_document: ` before chunk body text
- ask uses `search_query: ` before the question

Stored `policy_chunks.text` stays unprefixed. Only the embedding call sees the
prefix. After enabling this, the corpus must be re-ingested so every stored
vector was built with `search_document:`.

## Why

Nomic trained asymmetric retrieval modes. Queries and documents are different
shapes of text; the prefixes put them in the matching regions of the embedding
space and improve semantic ranking for paraphrased questions. Keyword `ILIKE`
search is unchanged and still uses the raw question and chunk text.

## What stays

- Hybrid merge, keyword terms, and generation prompts are unchanged.
- The minimal embed proof script uses the same prefixes.
