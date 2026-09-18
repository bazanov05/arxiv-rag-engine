from psycopg_pool import ConnectionPool
from google import genai
from dotenv import load_dotenv
import os

from normalizer_python import TextNormalizer
from src.model.encoder import PaperEncoder
from src.data.loader import retrieve_top_k_papers


load_dotenv()


class RAGEngine():
    """Orchestrates a sequential Retrieval-Augmented Generation (RAG) pipeline over an ArXiv database.

    This engine ties together text normalization, dense neural representation,
    vector similarity search in PostgreSQL, prompt synthesis, and generative answer 
    formulation using the Google GenAI SDK.
    """
    def __init__(
            self, 
            pool: ConnectionPool, 
            normalizer: TextNormalizer,
            encoder: PaperEncoder, 
            model_name: str = "gemini-2.5-flash"
        ):
        """Initializes the RAGEngine with database resources, neural encoders, and LLM clients.

        Args:
            pool (ConnectionPool): An active connection pool for PostgreSQL/pgvector queries.
            normalizer (TextNormalizer): A C++ accelerated preprocessor instance for cleaning text.
            encoder (PaperEncoder): PyTorch model used to project text into 256-dimensional embeddings.
            model_name (str, optional): Target Gemini model identifier used for response generation.
                Defaults to 'gemini-2.5-flash'.

        Raises:
            ValueError: If the `GEMINI_API_KEY` environment variable is not defined or is empty.
        """
        self._connection_pool = pool
        self._normalizer = normalizer
        self._model = encoder

        # get gemini api key from .env to create client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")

        self._client = genai.Client(api_key=api_key.strip())
        self._model_name = model_name
        
    def _format_prompt(self, user_query: str, retrieved_papers: list[dict]) -> str:
        """Formats the user query and retrieved papers into a structured prompt for the LLM.

        If no documents are found, it instructs the model to inform the user and
        propose reformulated queries. If documents exist, it formats paper metadata
        (including publication date and cosine similarity score) and enforces
        grounded generation with citations.

        Args:
            user_query (str): The raw question asked by the user.
            retrieved_papers (list[dict]): List of dictionaries containing:
            - paper_id (str)
            - title (str)
            - abstract (str)
            - categories (str)
            - published_date (Any)
            - cosine_similarity (float)

        Returns:
            str: Ready-to-send prompt string for the generation model.
        """
        if not retrieved_papers:
            return (
                "You are an expert academic research assistant.\n\n"
                f'The user asked: "{user_query}"\n\n'
                "Status: No matching papers were found in the local ArXiv database"
                " for this query.\n\n"
                "Instructions:\n"
                "1. Inform the user clearly that no relevant research papers were"
                " found.\n"
                "2. Suggest 2-3 reformulated, more specific search queries or"
                " keywords that could yield better retrieval results on this"
                " topic.\n"
                "3. Do not invent paper titles or fabricate facts."
            )

        # format retrieved papers into numbered source blocks
        formatted_sources = []
        for idx, paper in enumerate(retrieved_papers, start=1):
            paper_id = paper.get("paper_id", "N/A")
            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()
            categories = paper.get("categories", "General")
            published_date = paper.get("published_date", "Unknown date")
            similarity = paper.get("cosine_similarity", 0.0)

            entry = (
                f"[{idx}] Paper ID: {paper_id}\n"
                f"    Title: {title}\n"
                f"    Published: {published_date}\n"
                f"    Categories: {categories}\n"
                f"    Similarity Score: {similarity:.4f}\n"
                f"    Abstract: {abstract}"
            )
            formatted_sources.append(entry)

        context = "\n\n".join(formatted_sources)

        return (
            "You are an expert AI research assistant. Answer the user's research"
            " question using ONLY the retrieved ArXiv paper abstracts provided"
            " below.\n\n"
            "Guidelines:\n"
            "- Base your answer strictly on the provided context.\n"
            "- Use bracketed numerical citations (e.g., [1], [2]) directly after"
            " claims to cite sources.\n"
            "- If the retrieved papers do not contain sufficient information to"
            " answer the question fully, state explicitly what is missing.\n"
            "- Do not hallucinate external references or methods.\n\n"
            f"### User Question:\n{user_query}\n\n"
            f"### Retrieved Context:\n{context}\n\n"
            "### Grounded Response:"
        )

    def _retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Executes dense retrieval against the PostgreSQL vector store.

        Cleans the input query string via the normalizer, generates an embedding vector
        via the encoder model, and queries the database for the top-k most similar records
        using cosine similarity.

        Args:
            query (str): The raw search query or question.
            k (int, optional): The maximum number of nearest paper records to retrieve. Defaults to 5.

        Returns:
            list[dict]: A list of dictionary objects representing the top-k retrieved papers.
        """
        clean_query: str = self._normalizer.clean(query)
        search_vector: list[float] = self._model.generate_embeddings(texts=[clean_query])[0]

        with self._connection_pool.connection() as conn:
            top_k_papers: list[dict] = retrieve_top_k_papers(conn=conn, query_vector=search_vector, k=k)

        return top_k_papers

    def query(self, query_text: str, k: int = 5) -> tuple[str, list[dict]]:
        """Processes an end-to-end question answering request through the RAG pipeline.

        Retrieves relevant context records from the database, synthesizes an augmented
        instruction prompt, and invokes the Gemini model to produce a final cited response.

        Args:
            query_text (str): The research question submitted by the user.
            k (int, optional): The number of relevant papers to retrieve for context augmentation.
                Defaults to 5.

        Returns:
            str: The LLM-generated plain text response, complete with citations.
        """
        top_k_papers = self._retrieve(query=query_text, k=k)
        final_prompt = self._format_prompt(user_query=query_text, retrieved_papers=top_k_papers)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=final_prompt,
        )

        return (response.text or "", top_k_papers)