# Mini RAG Lab

A grounded employee expense-policy assistant. It chunks `policy.md`, embeds
each section, stores text plus vectors in PostgreSQL/pgvector, retrieves the
nearest sections, and answers only from that evidence.

The application is CLI-only.

## Submission files

| Requirement | What to submit |
| --- | --- |
| Application source | This repository, or a GitHub zip of it |
| Policy document | `policy.md` |
| Database schema | `migrations/001_create_policy_chunks.sql` and [docs/schema.md](docs/schema.md) |
| Ingestion command | `python -m mini_rag_lab ingest` in the application source |
| Run instructions | [docs/running.md](docs/running.md) |
| Six required questions | [docs/required-questions.md](docs/required-questions.md) |

## How the pipeline works

```text
policy.md
  -> structural split at ## headings (exactly 6 chunks)
  -> nomic-embed-text (768 dimensions)
  -> policy_chunks in PostgreSQL/pgvector

question
  -> embed the question
  -> cosine distance search, LIMIT 3, ascending
  -> generate from the nearest chunk only
  -> cite stored metadata, or refuse with no citation
```

## Quick start

See [docs/running.md](docs/running.md) for the full steps.

```shell
ollama pull nomic-embed-text
ollama pull qwen3:8b
docker compose up --build --detach
docker compose exec app bash
python -m mini_rag_lab migrate
python -m mini_rag_lab ingest
python -m mini_rag_lab evaluate
```

## Package layout

```text
src/mini_rag_lab/
├── cli.py            # migrate, ingest, ask, evaluate
├── config.py         # environment settings
├── runtime.py        # adapter and service wiring
├── migrations.py     # SQL migration runner
├── prompts/          # grounded generation system prompt
├── domain/           # models, policy parsing, ports
├── services/         # ingestion, retrieval, generation, query, evaluation
└── adapters/         # Ollama and PostgreSQL/pgvector
```
