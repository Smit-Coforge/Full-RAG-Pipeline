# Database schema

The assignment schema lives in `migrations/001_create_policy_chunks.sql`.
Apply it with:

```shell
python -m mini_rag_lab migrate
```

or:

```shell
mini-rag-lab migrate
```

## What is stored

Each row is one numbered section from `policy.md`. The table keeps the original
section text, the embedding vector, and the metadata needed for citations in
the same record.

| Column | Purpose |
| --- | --- |
| `chunk_id` | Stable ID, for example `expense-policy:v2.0:section-1` |
| `document` | Document title (`Employee Expense Policy`) |
| `version` | Document version (`2.0`) |
| `section` | Section number (`1` through `6`) |
| `section_title` | Section heading (`Meals`) |
| `text` | Original section body, not truncated |
| `embedding` | Full 768-dimension `nomic-embed-text` vector |
| `embedding_model` | Model that produced the vector |
| `created_at`, `updated_at` | Row timestamps |

`UNIQUE (document, version, section)` prevents duplicate sections for one
policy version. Re-running ingest replaces that version in one transaction and
leaves exactly six rows.

## Migration SQL

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_chunks (
    chunk_id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    version TEXT NOT NULL,
    section TEXT NOT NULL,
    section_title TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document, version, section)
);
```

Retrieval uses cosine distance:

```sql
ORDER BY embedding <=> :query_vector ASC
LIMIT 3;
```
