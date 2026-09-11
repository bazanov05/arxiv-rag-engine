CREATE TABLE papers (
    paper_id VARCHAR(32) PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    categories TEXT[] NOT NULL,
    published_date DATE,
    normalized_abstract TEXT,
    base_embedding vector(768) 
);