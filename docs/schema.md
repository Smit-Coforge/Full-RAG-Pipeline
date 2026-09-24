# Database schema

The schema lives in `migrations/001_create_policy_chunks.sql`.
Apply it with:

```shell
python -m mini_rag_lab migrate
```

or:

```shell
mini-rag-lab migrate
```

## What is stored

Each row is one policy section chunk (from `corpus/` PDF/DOCX by default, or
from an optional Markdown file via `ingest --policy file.md`). The table keeps
the section text, the embedding vector, and citation metadata in the same
record.

| Column | Purpose |
| --- | --- |
| `chunk_id` | Stable ID, for example `hr-policy:v2.0:section-7` |
| `document` | Document title (`HR Policy`) |
| `version` | Document version (`2.0`) |
| `section` | Section number (`7`) |
| `section_title` | Section heading (`Shared Refrigerator Policy`) |
| `text` | Section body used for generation and keyword search |
| `embedding` | 768-dimension `nomic-embed-text` vector (`search_document:` at embed time) |
| `embedding_model` | Model that produced the vector |
| `created_at`, `updated_at` | Row timestamps |

`chunk_id` is the primary key. Re-running ingest for one document version
replaces that version’s rows in one transaction. Other documents in the table
are left alone.

`UNIQUE (document, version, section)` assumes at most one row per section
number for a version. The chunker can emit `-part-2` ids if a section exceeds
the 512-token cap; those parts share the same `section` value and would
conflict with this unique constraint. The current corpus never hits that
split. If you need multi-part sections in the DB, widen the unique key (for
example include `chunk_id` only, or add a `part` column) in a follow-up
migration.

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

## Retrieval (application, not only SQL)

Vector branch (candidate pool):

```sql
ORDER BY embedding <=> :query_vector ASC
LIMIT 8;
```

Keyword branch uses `ILIKE` on `text` and `section_title`. Hybrid merges both
lists with RRF, then a CrossEncoder keeps the final top 3.
