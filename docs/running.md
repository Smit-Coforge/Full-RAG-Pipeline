# Running Mini RAG Lab

The application is a CLI. Docker runs the app and PostgreSQL/pgvector. Ollama
runs on the host machine.

## Prerequisites

- Docker Desktop
- [Ollama](https://ollama.com/) running on the host
- These local models:

```shell
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Confirm Ollama is reachable:

```shell
curl http://localhost:11434/api/tags
```

## Start the services

From the repository root:

```shell
docker compose up --build --detach
```

The app container reaches host Ollama at `http://host.docker.internal:11434`.

Enter the app container:

```shell
docker compose exec app bash
```

## Ingestion command

There is no separate ingest script to upload. Ingestion is this command, which
is part of the application source:

```shell
python -m mini_rag_lab migrate
python -m mini_rag_lab ingest
```

`migrate` applies `migrations/001_create_policy_chunks.sql`. `ingest` reads
`policy.md`, splits it into six sections, embeds each section, and stores the
rows in PostgreSQL.

Equivalent entry point after install:

```shell
mini-rag-lab migrate
mini-rag-lab ingest
```

Successful ingest prints:

```json
{"document": "Employee Expense Policy", "version": "2.0", "chunks_stored": 6}
```

## Ask a question

```shell
python -m mini_rag_lab ask "How much can I spend on food each day?"
```

The command prints the assignment JSON shape: `answer`, `citation`, and up to
three `retrieved_chunks` with numeric cosine distances in ascending order.

## Run the six required questions

```shell
python -m mini_rag_lab evaluate
```

Exit code `0` means all six questions passed. Saved output is in
`docs/required-questions.md`.

## Tests

```shell
python -m pytest tests/unit
RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration/test_database.py
RUN_LIVE_TESTS=1 python -m pytest tests/integration/test_live_pipeline.py
```

## Stop

```shell
docker compose down
```

The named PostgreSQL volume survives that command. Use
`docker compose down --volumes` only if you want to delete the database.
