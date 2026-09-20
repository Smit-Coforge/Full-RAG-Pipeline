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
