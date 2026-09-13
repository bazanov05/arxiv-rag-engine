from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class ArxivPaper:
    """Represents a research paper fetched for downstream ingestion."""
    paper_id: str
    title: str
    abstract: str
    categories: list[str]
    published_date: str
    normalized_title: str | None = None
    normalized_abstract: str | None = None
    base_embedding: list[float] | None = None
    keywords: list[str] | None = None


def fetch_arxiv_papers(
    search_query: str = "machine_learning",
    total_papers: int = 4000,
    batch_size: int = 100,
) -> list[ArxivPaper]:
    print(f"Starting ingestion: streaming {total_papers} papers from Hugging Face...")

    dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)

    papers: list[ArxivPaper] = []
    target_categories = {"cs.LG", "cs.CL", "cs.AI", "cs.CV", "stat.ML"}

    for item in dataset:
        title = " ".join(item.get("title", "").split())
        abstract = " ".join(item.get("abstract", "").split())
        paper_id = str(item.get("id", item.get("paper_id", "")))

        if not title or not abstract or not paper_id:
            continue

        raw_cats = item.get("categories", "")
        if isinstance(raw_cats, str):
            cats = raw_cats.split()
        elif isinstance(raw_cats, list):
            cats = raw_cats
        else:
            cats = []

        # Keep paper if any category intersects target categories (or if categories empty)
        if cats and not any(cat in target_categories for cat in cats):
            continue

        # Extract date safely
        pub_date = item.get("update_date") or item.get("published") or "2024-01-01"
        if len(str(pub_date)) == 4:
            pub_date = f"{pub_date}-01-01"

        papers.append(
            ArxivPaper(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                categories=cats if cats else ["cs.AI"],
                published_date=str(pub_date)[:10],
            )
        )

        if len(papers) % 500 == 0 and len(papers) > 0:
            print(f"Loaded {len(papers)} / {total_papers} papers...")

        if len(papers) >= total_papers:
            break

    print(f"Ingestion complete: retrieved {len(papers)} valid papers.")
    return papers