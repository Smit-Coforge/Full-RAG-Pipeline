# ADR 0001: Keep PostgreSQL with pgvector

- Date: 2026-09-22

## Decision

The vector store stays PostgreSQL with pgvector. Chroma is not adopted, neither as a library inside the app process nor as its own container.

## Why

The assignment lists Chroma as one suggested store. The lab already stores policy chunks in pgvector, keeps each document version, cites the section a chunk came from, and has tests around that path. Search uses the same embeddings and the same cosine comparison either way, so a new database would not make retrieval better.

Embedded Chroma would remove the Postgres service. Compose would run only the app container, data would sit in a local directory, and there would be no migration step. Ollama would stay on the host, the same as it does now. A separate Chroma server container was not considered.

That simpler layout is not worth the rewrite. Moving stores means replacing the repository, the SQL migrations, Compose, and the database tests. Replacing one policy version is already one transaction in `replace_document`. With Chroma, upsert by `chunk_id` still needs that same "replace this version only" behavior written again. Hybrid search and reranking are application code on top of whichever store is used.

Postgres also makes the saved chunks easy to inspect with SQL. It is more database than a local lab strictly needs, and integration tests still require the database container. Those costs are already paid.

## What stays

- Compose runs the app and Postgres.
- Ingest still uses `DATABASE_URL` and `python -m mini_rag_lab migrate`.
- v1 and v2 of the same policy remain separate rows because their chunk ids differ.
- Later hybrid retrieval and reranking are added in application code, not by changing the database.
