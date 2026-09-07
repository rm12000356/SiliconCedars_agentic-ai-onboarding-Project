-- Enable pgvector before anything else in this file can use the vector type.
CREATE EXTENSION IF NOT EXISTS vector;

-- General-access embeddings. Same access tier as lessons_learned/employees/sales:
-- general_role can read this. No sensitive content is embedded here.
-- A future sensitive_embeddings table (not granted to general_role at all,
-- same invisibility property as salaries/credentials) can be added later
-- if sensitive documents ever need to be embedded.
CREATE TABLE general_embeddings (
    id SERIAL PRIMARY KEY,
    source_table TEXT NOT NULL,      -- e.g. 'lessons_learned', future 'documents'
    source_id INTEGER NOT NULL,      -- the row id this chunk came from
    content TEXT NOT NULL,           -- the actual text that was embedded
    embedding vector(384),           -- dimension depends on the embedding model chosen
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

GRANT SELECT ON general_embeddings TO general_role;