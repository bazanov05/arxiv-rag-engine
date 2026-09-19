import os
from dotenv import load_dotenv
from google import genai
from psycopg_pool import ConnectionPool

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
            model_name: str = "gemini-3.6-flash"
        ):
        """Initializes the RAGEngine with database resources, neural encoders, and LLM clients.

        Args:
            pool (ConnectionPool): An active connection pool for PostgreSQL/pgvector queries.
            normalizer (TextNormalizer): A C++ accelerated preprocessor instance for cleaning text.
            encoder (PaperEncoder): PyTorch model used to project text into 256-dimensional embeddings.
            model_name (str, optional): Target Gemini model identifier used for response generation.
                Defaults to 'gemini-3.6-flash'.

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
        
        # agent state tracking (Required for the SDK tool loop)
        self._search_attempts: int = 0
        self._current_k: int = 5
        self._last_papers: list[dict] = []
        
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

    def _agent_search(self, search_query: str) -> str:
        """Searches the ArXiv database for academic papers.
        
        If the search results are irrelevant or empty, you can call this tool again 
        with a reformulated, broader, or more specific query.
        """
        self._search_attempts += 1
        
        if self._search_attempts > 3:
            return (
                "SYSTEM OVERRIDE: Maximum search limit (3) reached. "
                "You are forbidden from searching again. Answer the user's question "
                "using ONLY the information you have gathered so far. If you still "
                "lack the info, state exactly what is missing."
            )

        print(f"  [Agent] Searching DB for: '{search_query}' (Attempt {self._search_attempts}/3)")
        
        # run your existing retrieval
        papers = self._retrieve(query=search_query, k=self._current_k)
        
        # save papers for CLI `/sources` command
        self._last_papers = papers 
        
        # format the results and hand them back to the LLM
        return self._format_prompt(user_query=search_query, retrieved_papers=papers)

    def query(self, query_text: str, k: int = 5) -> tuple[str, list[dict]]:
        """Processes an end-to-end question answering request through the RAG pipeline.

        Retrieves relevant context records from the database, synthesizes an augmented
        instruction prompt, and invokes the Gemini model to produce a final cited response.

        Args:
            query_text (str): The research question submitted by the user.
            k (int, optional): The number of relevant papers to retrieve for context augmentation.
                Defaults to 5.

        Returns:
            tuple[str, list[dict]]: The LLM-generated plain text response with citations, 
                and the list of retrieved papers.
        """
        # stash the state so the SDK tool (_agent_search) can access it
        self._search_attempts = 0
        self._current_k = k
        self._last_papers = []

        # start an autonomous chat session
        chat = self._client.chats.create(
            model=self._model_name,
            config=dict(
                tools=[self._agent_search],
                system_instruction=(
                    "You are an autonomous research agent. If the user asks a question, "
                    "use your _agent_search tool to find papers. If the returned papers do not "
                    "answer the question, REWRITE the query using different keywords and "
                    "search again. Provide a final cited answer once you have good data."
                ),
                temperature=0.1
            )
        )

        response = chat.send_message(query_text)

        # return the LLM's text and the papers we stashed during the tool call
        return (response.text or "", self._last_papers)