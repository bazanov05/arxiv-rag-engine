import numpy as np
import json
import math
from psycopg.rows import dict_row

from src.data.arxiv_fetcher import ArxivPaper


def load_data_to_db(conn, papers: list[ArxivPaper]) -> None:
    """
    Bulk inserts ArXiv paper records into the PostgreSQL 'papers' table.

    Uses the PostgreSQL COPY FROM STDIN protocol via psycopg for high-performance 
    ingestion. Safely formats Python lists into PostgreSQL TEXT[] arrays and 
    handles None values for embeddings to ensure SQL NULL compatibility.

    Args:
        conn: An active psycopg database connection object.
        papers (list[ArxivPaper]): A list of populated ArxivPaper dataclass instances.
    """
    copy_query = """
        COPY papers (paper_id, title, normalized_title, abstract, categories, published_date, normalized_abstract, base_embedding, keywords) 
        FROM STDIN
    """

    with conn.cursor() as cursor:
        with cursor.copy(copy_query) as copy:
            for paper in papers:
                # safely cast vector to string, or keep as None for SQL NULL
                pg_embedding = str(paper.base_embedding) if paper.base_embedding else None
                
                copy.write_row( 
                    (
                        paper.paper_id,
                        paper.title,
                        paper.normalized_title,
                        paper.abstract,
                        paper.categories,
                        paper.published_date,
                        paper.normalized_abstract,
                        pg_embedding,
                        paper.keywords
                    )
                )
    
    conn.commit()


def init_db(conn, schema_path: str = "src/db/schema.sql") -> None:
    """
    Reads and executes the SQL schema to create necessary database tables.

    Args:
        conn: An active psycopg database connection object.
        schema_path (str): The file path to the SQL schema definition.
    """
    with open(file=schema_path, mode="r", encoding="utf-8") as f:
        schema_sql = f.read()

    with conn.cursor() as cursor:
        cursor.execute(schema_sql)
    
    conn.commit()


def create_index(
    conn, 
    num_of_probes: int = 0, 
    lists: int = 0, 
    column: str = "base_embedding"
) -> None:
    """
    Creates an IVFFlat index on the specified vector column and sets search probes.

    IVFFlat clusters vectors into lists via k-means at index creation time. When querying,
    it restricts the similarity search exclusively to the nearest centroids determined by
    `ivfflat.probes`.

    If `lists` is <= 0, it dynamically counts table rows (N) and assigns lists = ceil(sqrt(N))
    (with a floor of 10). If `num_of_probes` is <= 0, it defaults to max(1, ceil(sqrt(lists))).

    Args:
        conn: An active psycopg database connection object.
        num_of_probes (int, optional): Number of nearest centroid clusters to inspect 
                                       at query time. Higher values increase Recall@k 
                                       at the cost of query latency. If <= 0, defaults 
                                       to ceil(sqrt(lists)).
        lists (int, optional): Number of k-means clusters to create at index build time.
                               If <= 0, defaults to ceil(sqrt(table_size)).
        column (str, optional): Target vector column to index. Defaults to "base_embedding".
    """
    with conn.cursor() as cursor:
        # dynamically determine lists via sqrt(N) if lists <= 0
        if lists <= 0:
            cursor.execute(f"SELECT COUNT(*) FROM papers WHERE {column} IS NOT NULL;")
            total_rows = cursor.fetchone()[0]
            # sqrt rule of thumb with a practical lower bound of 10
            lists = max(10, math.ceil(math.sqrt(total_rows))) if total_rows > 0 else 65

        # dynamically set probes via sqrt(lists) if probes <= 0
        if num_of_probes <= 0:
            num_of_probes = max(1, math.ceil(math.sqrt(lists)))

        index_name = (
            "pretrained_embedding_idx" 
            if column == "base_embedding" 
            else "finetuned_embedding_idx"
        )

        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON papers USING ivfflat ({column} vector_cosine_ops) "
            f"WITH (lists = {lists});"
        )
        
        cursor.execute(
            "SELECT set_config('ivfflat.probes', %s, false);",
            (str(num_of_probes),)
        )
    
    conn.commit()


def build_distance_matrix(conn, column: str = "base_embedding") -> dict[str, dict[int, float]]:
    """
    Builds an in-memory pairwise cosine distance matrix for all papers.

    Fetches embeddings from the database and computes pairwise distances using 
    matrix multiplication. Assumes the embedding vectors are already L2-normalized 
    by the model, allowing dot product to act as cosine similarity. Used for 
    efficient semi-hard negative mining during Triplet Loss training.

    Args:
        conn: An active psycopg database connection object.
        column (str): The target embedding column to pull vectors from.

    Returns:
        A nested dictionary mapping each paper_id to a dictionary of distances 
        to all other paper_ids. Example: {'2305.1234': {'2306.5678': 0.15, ...}}
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT " 
            "   paper_id, " 
            f"   {column} " 
            "FROM papers " 
            f"WHERE {column} IS NOT NULL " 
            "ORDER BY paper_id ASC;"
        )

        result = cursor.fetchall()  # result is list of tuples(paper_id, base_embedding)

        if not result:
            raise ValueError(f"No embeddings found in column '{column}'. Run main.py first.")

        # Postgres might return embeddings as strings
        # so we cannot directly map them to np.array
        ids = [row[0] for row in result]
        raw_embeddings = [row[1] for row in result]

        # parse string representation into float matrix
        if isinstance(raw_embeddings[0], str):
            embeddings = np.array([json.loads(vec) for vec in raw_embeddings], dtype=np.float32)
        else:
            embeddings = np.array(raw_embeddings, dtype=np.float32)

        # get the similarities between vectors (1000, 384) @ (384, 1000) = (1000, 1000)
        # assumes vectors are normalized (len == 1) so cosine similarity = dot product
        similarities = embeddings @ embeddings.T
        cosine_distances = np.subtract(1, similarities) # cosine_distace = 1 - cosine_similarity

        distance_matrix = {}

        # build a result dict - {id1: {id2: distance}}
        for i, id1 in enumerate(ids):
            distance_matrix[id1] = {}

            for j, id2 in enumerate(ids):
                distance_matrix[id1][id2] = cosine_distances[i][j]

        return distance_matrix


def fetch_papers_embeddings(conn, column: str = "base_embedding") -> dict[str, list[float]]:
    """
    Retrieves all non-null paper embeddings from the database.

    Args:
        conn: An active PostgreSQL database connection.
        column (str): The specific embedding column to fetch.

    Returns:
        A dictionary mapping each paper_id to its embedding vector (list of floats).
    """
    with conn.cursor() as cursor:
        cursor.row_factory = dict_row

        cursor.execute(
            f"SELECT paper_id, {column} FROM papers "
            f"WHERE {column} IS NOT NULL "
            "ORDER BY paper_id;"
        )
        
        results = cursor.fetchall()
        
        embeddings_dict = {}
        for row in results:
            raw_vec = row[f"{column}"]
            # parse string representation into list of floats
            if isinstance(raw_vec, str):
                embeddings_dict[row["paper_id"]] = json.loads(raw_vec)
            else:
                embeddings_dict[row["paper_id"]] = list(raw_vec)

        return embeddings_dict


def save_idf_scores(conn, idf_scores: dict[str, float]) -> None:
    """
    Bulk inserts vocabulary IDF scores into the PostgreSQL 'idf_scores' table.

    Uses the PostgreSQL COPY FROM STDIN protocol via psycopg for fast bulk insertion.

    Args:
        conn: An active psycopg database connection object.
        idf_scores: Dictionary mapping vocabulary words to their IDF weights.

    Returns:
        None.
    """
    copy_query = """
        COPY idf_scores (word, idf_score) 
        FROM STDIN
    """
    with conn.cursor() as cursor:
        with cursor.copy(copy_query) as copy:
            for word, score in idf_scores.items():
                copy.write_row((word, score))

    conn.commit()


def fetch_idf_scores(conn) -> dict[str, float]:
    """
    Retrieves all vocabulary words and their corresponding IDF weights from the database.

    Args:
        conn: An active psycopg database connection object.

    Returns:
        Dictionary mapping vocabulary terms to their stored IDF scores.
    """
    with conn.cursor() as cursor:
        cursor.execute("SELECT word, idf_score FROM idf_scores;")
        return {row[0]: float(row[1]) for row in cursor.fetchall()}


def fetch_paper_keywords(conn) -> list[list[str]]:
    """
    Fetches extracted keyword lists for all papers ordered by paper_id.

    Args:
        conn: An active psycopg database connection object.

    Returns:
        List of keyword lists corresponding to papers ordered by paper_id.
    """
    with conn.cursor() as cursor:
        cursor.execute("SELECT keywords FROM papers ORDER BY paper_id ASC;")
        return [row[0] if row[0] is not None else [] for row in cursor.fetchall()]


def retrieve_top_k_papers(
    conn, 
    query_vector: list[float], 
    k: int = 5, 
    column: str = "finetuned_embedding"
) -> list[dict]:
    """
    Retrieves the top-k most semantically relevant papers using vector cosine distance.

    Args:
        conn: An active psycopg database connection object.
        query_vector: 256-dimensional L2-normalized embedding of the user's query.
        k: Number of nearest neighbors to return.
        column: The database vector column to compare against.

    Returns:
        List of dictionaries containing paper metadata and similarity scores.
    """
    query = f"""
        SELECT 
            paper_id, 
            title, 
            abstract, 
            categories,
            published_date,
            1 - ({column} <=> %s::vector) AS cosine_similarity
        FROM papers
        WHERE {column} IS NOT NULL
        ORDER BY {column} <=> %s::vector ASC
        LIMIT %s;
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (str(query_vector), str(query_vector), k))
        rows = cursor.fetchall()

    return [
        {
            "paper_id": r[0],
            "title": r[1],
            "abstract": r[2],
            "categories": r[3],
            "published_date": r[4],
            "cosine_similarity": float(r[5])
        }
        for r in rows
    ]
