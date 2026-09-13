from src.data.loader import (
    init_db, 
    load_data_to_db,
    save_idf_scores
)
from src.data.arxiv_fetcher import fetch_arxiv_papers, ArxivPaper
from src.db import connection
from src.model.embedder import PaperEmbedder
from normalizer_python import TextNormalizer
from tfidf_python import TfidfKeywordExtractor


PATH_TO_SQL_SCHEMA = "./src/db/schema.sql"
PATH_TO_STOP_WORDS = "./normalizer/data/stopwords.txt"


def _fetch_stop_words(path: str = PATH_TO_STOP_WORDS) -> set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            stop_words = set()

            for line in f:
                cleaned_word = line.strip()

                if cleaned_word:
                    stop_words.add(cleaned_word)

            return stop_words
    except FileNotFoundError:
        raise FileNotFoundError(f"Path {path} does not exist")
    except IsADirectoryError:
        raise IsADirectoryError("You are trying to open the dir, not file")
    except PermissionError:
        raise PermissionError("You do not have permission to open this file")


def setup():
    connection.init_pool()

    # fetch arxiv_papers from public API
    arxiv_papers: list[ArxivPaper] = fetch_arxiv_papers()

    # fetch stop words from txt file and init text normalizer
    stop_words: set[str] = _fetch_stop_words()
    normalizer = TextNormalizer(stop_words)  

    # clean every abstract and title
    for paper in arxiv_papers:
        paper.normalized_abstract = normalizer.clean(paper.abstract)
        paper.title = normalizer.clean(paper.title)
        

    # batch generate 768-dim base embeddings (Title + Abstract)
    texts_to_embed = [f"{p.title} {p.normalized_abstract}" for p in arxiv_papers]
    embedder = PaperEmbedder()
    embeddings: list[list[float]] = embedder.generate_embeddings(texts=texts_to_embed)

    for paper, emb in zip(arxiv_papers, embeddings):
        paper.base_embedding = emb

    # for every paper extract top 10 most rare+frequent words from Title + Abstract sections
    extractor = TfidfKeywordExtractor(papers=texts_to_embed, top_k=10)
    keywords = extractor.extract_keywords()

    for paper, top_k_keywords in zip(arxiv_papers, keywords):
        paper.keywords = top_k_keywords

    # the dict of words and their idf scores
    # only top_k words from every text are considered
    # used later in torch dataset for similairty score
    # used for calculation of global idf mean score
    idf_scores: dict[str, float] = extractor.get_idf_map()

    with connection.pool.connection() as conn:
        # initialize db based on schema and load  data to it
        init_db(conn=conn, schema_path=PATH_TO_SQL_SCHEMA)
        load_data_to_db(conn=conn, papers=arxiv_papers)
        save_idf_scores(conn=conn, idf_scores=idf_scores)

    connection.close_pool()


if __name__ == "__main__":
    setup()
