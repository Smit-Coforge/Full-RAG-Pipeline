# Mini RAG Lab

A grounded policy RAG assistant. It ingests PDF/DOCX policies from `corpus/`,
embeds sections with Ollama `nomic-embed-text`, stores text plus vectors in
PostgreSQL/pgvector, retrieves with hybrid search (cosine + `ILIKE`), merges
with RRF, reranks with a MiniLM CrossEncoder, and answers only from that
evidence.

The application is CLI-only. An optional Markdown ingest path still exists for
a single six-section `##` policy file; the default product path is `corpus/`.

## Submission / lab notes

| Topic | Location |
| --- | --- |
| Application source | This repository |
| Policy corpus | `corpus/` (PDF and DOCX) |
| Database schema | `migrations/001_create_policy_chunks.sql` and [docs/schema.md](docs/schema.md) |
| Architecture decisions | [docs/adr/](docs/adr/) |
| Run instructions | [docs/running.md](docs/running.md) |
| Legacy six-question evaluate | [docs/required-questions.md](docs/required-questions.md) (first mini-lab harness; to be replaced by the new eval suite) |

## How the pipeline works

```text
corpus/*.pdf|docx
  -> extract text
  -> split on top-level numbered sections (512-token / 80 overlap fallback)
  -> nomic-embed-text with search_document: prefix (768d)
  -> policy_chunks in PostgreSQL/pgvector

question
  -> retrieval strategy: vector | keyword | hybrid (CLI --strategy, or Jev via --router)
  -> vector: search_query: embed + cosine top 8
  -> keyword: ILIKE on text/title
  -> hybrid: RRF merge (k=60)
  -> CrossEncoder ms-marco-MiniLM-L-6-v2 -> top 3
  -> generate with qwen3:8b from those excerpts
  -> cite stored metadata, or refuse with no citation
```

`--strategy` is unchanged. Add `--router` only when you want Jev to pick the
path (needs `TYPESAFE_API_KEY` in `.env`).

## Quick start

See [docs/running.md](docs/running.md) for the full steps.

```shell
ollama pull nomic-embed-text
ollama pull qwen3:8b
docker compose up --build --detach
docker compose exec app bash
python -m mini_rag_lab migrate
python -m mini_rag_lab ingest
python -m mini_rag_lab ask "What does section 7.1 say about the refrigerator?"
```

## Package layout

```text
src/mini_rag_lab/
├── cli.py            # migrate, ingest, ask, evaluate
├── config.py         # environment settings
├── runtime.py        # adapter and service wiring
├── migrations.py     # SQL migration runner
├── prompts/          # grounded generation system prompt
├── domain/           # models, chunking, ports
├── services/         # ingestion, retrieval, generation, query, evaluation
└── adapters/         # Ollama, pgvector, CrossEncoder
```
