import torch

from src.db import connection
from src.data.loader import create_index, fetch_papers_embeddings
from src.model.encoder import PaperEncoder


WEIGHTS_PATH = "./src/model/data/weights.pt"


def update_finetuned_embeddings(conn, embeddings_dict: dict[str, list[float]], dim: int) -> None:
    """
    Adds the finetuned embedding column if missing and updates records by paper ID.

    Ensures the target vector column exists with the specified dimension, iterates
    over the dictionary of pre-computed projected vectors, and executes SQL UPDATEs
    per record.

    Args:
        conn: An active psycopg database connection object.
        embeddings_dict (dict[str, list[float]]): Mapping of paper_id to projected vector floats.
        dim (int): Dimensionality of the target vector representation (e.g., 256).
    """
    create_query = f"ALTER TABLE papers ADD COLUMN IF NOT EXISTS finetuned_embedding vector({dim});"
    update_query = "UPDATE papers SET finetuned_embedding = %s WHERE paper_id = %s"
    
    with conn.cursor() as cursor:
        cursor.execute(create_query)
        for paper_id, vector in embeddings_dict.items():
            cursor.execute(update_query, (str(vector), paper_id))
    conn.commit()


def embed() -> None:
    """
    Projects stored 768-dim base embeddings to fine-tuned vector space and indexes them.

    Loads the projection checkpoint, dynamically infers the target projection dimension
    from the saved state dictionary weights, instantiates PaperEncoder, and fetches
    existing base embeddings from the database. Passes the full batch through the linear
    projection head under torch.no_grad(), saves the resulting normalized embeddings into
    the 'finetuned_embedding' column, and builds an IVFFlat index with dynamic probes
    and list clustering.

    Raises:
        FileNotFoundError: If the projection weights checkpoint cannot be found at WEIGHTS_PATH.
        ValueError: If no base embeddings are present in the database to project.
    """
    connection.init_pool()

    # guard check: see if column exists and already has embeddings
    with connection.pool.connection() as conn:
        with conn.cursor() as cur:
            # verify if column exists before checking contents
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'papers' AND column_name = 'finetuned_embedding'
                );
            """)
            column_exists = cur.fetchone()[0]

            if column_exists:
                cur.execute("SELECT COUNT(*) FROM papers WHERE finetuned_embedding IS NOT NULL;")
                count = cur.fetchone()[0]
                if count > 0:
                    print(f"Database already contains {count} fine-tuned embeddings. Skipping embedding step.")
                    connection.close_pool()
                    return
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint["projection_state_dict"]

    # etract output projection dimension dynamically from weight tensor shape [out_features, in_features]
    projection_dim = state_dict["weight"].shape[0]

    # instantiate model with detected dimension and apply weights
    model = PaperEncoder(projection_dim=projection_dim)
    model._projection.load_state_dict(state_dict)
    model.eval()

    with connection.pool.connection() as conn:
        base_embeddings_dict = fetch_papers_embeddings(conn=conn)

        paper_ids = list(base_embeddings_dict.keys())
        raw_vectors = list(base_embeddings_dict.values())

        # project to 256-dim in one fast batch
        print(f"Projecting {len(paper_ids)} vectors to {projection_dim} dimensions...")
        with torch.no_grad():
            tensor_batch = torch.tensor(raw_vectors).to(device)
            # this calls model.forward(), which applies projection and L2 normalization
            projected_tensor = model(tensor_batch)

        projected_lists = projected_tensor.cpu().tolist()
        finetuned_embeddings_dict = {
            pid: vec for pid, vec in zip(paper_ids, projected_lists)
        }

        print("Saving finetuned embeddings to database...")
        update_finetuned_embeddings(conn, finetuned_embeddings_dict, dim=projection_dim)
        print("Creating IVFF indices...")

        lists = int(len(projected_lists) ** 0.5)
        probes = int(lists ** 0.5)
        create_index(conn=conn, num_of_probes=probes, lists=lists, column="finetuned_embedding")

    connection.close_pool()


if __name__ == "__main__":
    embed()
