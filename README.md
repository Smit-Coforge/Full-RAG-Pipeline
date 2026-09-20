# Mini RAG Lab

A grounded employee expense-policy assistant. It chunks `policy.md`, embeds
each section, stores text plus vectors in PostgreSQL/pgvector, retrieves the
nearest sections, and answers only from that evidence.

The application is CLI-only. There is no HTTP API.

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

Unsupported questions, including gym memberships, return exactly:

```text
The provided policy does not answer this question.
```

with `"citation": null`.

## Assignment deliverables

| Requirement | Location |
| --- | --- |
| Application source | `src/mini_rag_lab/` |
| Policy document | `policy.md` |
| Database schema | `migrations/001_create_policy_chunks.sql` |
| Ingestion command | `python -m mini_rag_lab ingest` |
| Run instructions | this README |
| Six required questions | `python -m mini_rag_lab evaluate` and `tests/integration/test_live_pipeline.py` |

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

## Schema

Each stored chunk is one numbered policy section:

```text
chunk_id, document, version, section, section_title,
text, embedding vector(768), embedding_model, timestamps
```

Example ID: `expense-policy:v2.0:section-1`. Re-running ingest replaces that
document version in one transaction and leaves six rows.

## Setup

Prerequisites:

- Docker Desktop must be running.
- Ollama must be running natively on macOS.
- Install the local models from a Mac terminal:

```shell
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Copy `.env.example` if you need local overrides. Compose already sets
`DATABASE_URL` and `OLLAMA_HOST=http://host.docker.internal:11434`.

```shell
docker compose up --build --detach
```

In Cursor, select **Dev Containers: Attach to Running Container...**, choose
`mini-rag-lab-app-1`, and select `/usr/local/bin/python` as the Python
interpreter. Run **Developer: Reload Window** after attaching.

```shell
python -m mini_rag_lab --help
curl http://host.docker.internal:11434/api/tags
```

## Commands

Run these from the app container after PostgreSQL is healthy:

```shell
python -m mini_rag_lab migrate
python -m mini_rag_lab ingest
python -m mini_rag_lab ask "How much can I spend on food each day?"
python -m mini_rag_lab evaluate
```

`ask` prints the assignment JSON shape:

```json
{
  "answer": "Employees may claim up to $65 per day for meals while traveling overnight.",
  "citation": {
    "document": "Employee Expense Policy",
    "version": "2.0",
    "section": "1. Meals"
  },
  "retrieved_chunks": [
    {"section": "1. Meals", "distance": 0.08}
  ]
}
```

`retrieved_chunks` contains at most three rows, sorted by ascending cosine
distance. `evaluate` runs all six required questions and exits `0` only when
every check passes.

## Required questions

| Question | Expected citation |
| --- | --- |
| How much can I spend on food each day? | 1. Meals |
| Can I book first-class airfare? | 3. Airfare |
| My hotel costs $250. What do I need? | 2. Hotels |
| Do I need a receipt for a $20 taxi? | 5. Receipts |
| Can I claim a limousine upgrade? | 4. Ground Transportation |
| Does the company reimburse gym memberships? | none |

## Tests

From the app container:

```shell
python -m pytest tests/unit
RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration/test_database.py
RUN_LIVE_TESTS=1 python -m pytest tests/integration/test_live_pipeline.py
python -m ruff check src tests
```

Default `pytest` stays fast: database and Ollama tests are skipped unless those
environment variables are set.

## Models

```text
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
GENERATION_MODEL=qwen3:8b
GENERATION_THINKING=false
```

The comparison that selected these defaults is in `docs/model-comparison.md`.

Stop the services with `docker compose down`. The named PostgreSQL volume
survives that command. Avoid `docker compose down --volumes` unless you intend
to delete the database.
