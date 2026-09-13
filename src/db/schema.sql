CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    paper_id VARCHAR(64) PRIMARY KEY,
    title TEXT NOT NULL,
    normalized_title TEXT,
    abstract TEXT NOT NULL,
    categories TEXT[] NOT NULL,
    published_date DATE,
    normalized_abstract TEXT,
    base_embedding vector(768),
    keywords TEXT[]
);

CREATE TABLE IF NOT EXISTS idf_scores (
    word TEXT PRIMARY KEY,
    idf_score FLOAT NOT NULL
);