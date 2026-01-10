-- Schema for code semantic search with pgvector
-- Run this on your PostgreSQL database after enabling pgvector

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main table for code chunks
CREATE TABLE IF NOT EXISTS code_embeddings (
    id SERIAL PRIMARY KEY,

    -- Source information
    repo_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,

    -- Code content
    chunk_type VARCHAR(50) NOT NULL,  -- 'function', 'class', 'file', 'docstring'
    name VARCHAR(255),                 -- function/class name if applicable
    content TEXT NOT NULL,             -- actual code or text
    start_line INTEGER,
    end_line INTEGER,

    -- Embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding vector(384) NOT NULL,

    -- Metadata
    language VARCHAR(50),
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Prevent duplicates
    UNIQUE(repo_name, file_path, chunk_type, name, start_line)
);

-- Index for fast similarity search (HNSW - better recall, good for all dataset sizes)
-- m: max connections per layer (16 is a good default)
-- ef_construction: size of dynamic candidate list during construction (64-200)
CREATE INDEX IF NOT EXISTS code_embeddings_vector_hnsw_idx
ON code_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index for filtering by repo
CREATE INDEX IF NOT EXISTS code_embeddings_repo_idx
ON code_embeddings(repo_name);

-- Index for filtering by language
CREATE INDEX IF NOT EXISTS code_embeddings_language_idx
ON code_embeddings(language);

-- Example search query:
-- SELECT file_path, name, content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
-- FROM code_embeddings
-- WHERE repo_name = 'my-repo'
-- ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 10;
