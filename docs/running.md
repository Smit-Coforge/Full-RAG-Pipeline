# Running Mini RAG Lab

The application is a CLI. Docker runs the app and PostgreSQL/pgvector. Ollama
runs on the host machine. The default corpus is the PDF/DOCX files under
`corpus/`.

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

## Ingestion

```shell
python -m mini_rag_lab migrate
python -m mini_rag_lab ingest
```

`migrate` applies `migrations/001_create_policy_chunks.sql`.

`ingest` defaults to `--policy corpus`. Every `.pdf` / `.docx` in that directory
is chunked, embedded with a `search_document:` prefix, and upserted. Re-ingest
replaces each document version; it does not delete unrelated documents left
from an older ingest.

Optional Markdown path (exactly six `##` sections, legacy mini-lab shape):

```shell
python -m mini_rag_lab ingest --policy path/to/policy.md
```

Successful corpus ingest prints a multi-document report, for example:

```json
{"documents": [{"document": "HR Policy", "version": "1.0", "chunks_stored": 9}], "chunks_stored": 63}
```

Equivalent entry point after install:

```shell
mini-rag-lab migrate
mini-rag-lab ingest
```

## Ask a question

```shell
python -m mini_rag_lab ask "What does section 7.1 say about the refrigerator?"
python -m mini_rag_lab ask "What does section 7.1 say about the refrigerator?" --strategy vector
python -m mini_rag_lab ask "What does section 7.1 say about the refrigerator?" --strategy keyword
python -m mini_rag_lab ask "What does section 7.1 say about the refrigerator?" --router
```

Default `--strategy` is `hybrid`: cosine + `ILIKE`, RRF merge, then MiniLM
CrossEncoder top 3. The JSON includes `answer`, `citation`,
`retrieved_chunks` (with `distance` and `rerank_score`), and
`retrieval_strategy`.

`--router` turns on the Jev strategy router (TypeSafe System One). Put your
key in a repo-root `.env` as `TYPESAFE_API_KEY=apikey_...` (see
`.env.example`). When `--router` is set, Jev picks `vector` | `keyword` |
`hybrid` and `--strategy` is ignored. Without `--router`, behavior is
unchanged.

First ask in a process loads the CrossEncoder weights into memory (and may
show a Hugging Face Hub warning without `HF_TOKEN`).

## Legacy evaluate (six expense-policy questions)

```shell
python -m mini_rag_lab evaluate
```

This still runs the first mini-lab meal/airfare/hotel cases against whatever
is in the database. It will not match the Doofenshmirtz corpus. Saved output
from that harness is in `docs/required-questions.md`. A new ≥8-question
corpus harness replaces this in a later lab part.

## Tests

```shell
python -m pytest tests/unit
RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration/test_database.py
RUN_LIVE_TESTS=1 python -m pytest tests/integration/test_live_pipeline.py
```

## Minimal embed proof

```shell
python scripts/minimal_embed_retrieve.py
```

## Stop

```shell
docker compose down
```

The named PostgreSQL volume survives that command. Use
`docker compose down --volumes` only if you want to delete the database.
