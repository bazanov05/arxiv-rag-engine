from dataclasses import dataclass
import time
import urllib.request
import xml.etree.ElementTree as ET


@dataclass
class ArxivPaper:
    """Represents a research paper fetched from the ArXiv API."""
    paper_id: str
    title: str
    abstract: str
    categories: list[str]
    published_date: str
    normalized_abstract: str | None = None
    base_embedding: list[float] | None = None


def fetch_arxiv_papers(
    search_query: str = "cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.AI",
    total_papers: int = 4000,
    batch_size: int = 500,
) -> list[ArxivPaper]:
    """
    Fetches research papers from the ArXiv API with automatic pagination and rate-limiting.

    Args:
        search_query: ArXiv search query. Formatted with + instead of spaces. 
                      Defaults to Machine Learning (cs.LG), NLP (cs.CL), and AI (cs.AI).
        total_papers: Total number of papers to retrieve.
        batch_size: Number of records to fetch per HTTP request (ArXiv recommends <= 1000).

    Returns:
        A list of ArxivPaper instances.
    """
    base_url = "http://export.arxiv.org/api/query?"
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[ArxivPaper] = []

    print(f"Starting ingestion: fetching {total_papers} papers matching '{search_query}'...")

    for start_idx in range(0, total_papers, batch_size):
        current_batch_size = min(batch_size, total_papers - start_idx)

        # build query manually to avoid urlencode escaping the '+' signs
        # which ArXiv uses as spaces in its query syntax.
        query_string = (
            f"search_query={search_query}"
            f"&start={start_idx}"
            f"&max_results={current_batch_size}"
            f"&sortBy=submittedDate"
            f"&sortOrder=descending"
        )
        url = base_url + query_string

        try:
            req = urllib.request.Request(
                url, 
                headers={"User-Agent": "SemanticResearchAssistant/1.0 (academic research)"}
            )
            with urllib.request.urlopen(req) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", ns)

            if not entries:
                print(f"No more records returned at index {start_idx}. Stopping.")
                break

            for entry in entries:
                # ArXiv ID format: 'http://arxiv.org/abs/2305.12345v1' -> '2305.12345'
                raw_id = entry.find("atom:id", ns).text.strip()
                paper_id = raw_id.split("/abs/")[-1].split("v")[0]

                title_elem = entry.find("atom:title", ns)
                title = " ".join(title_elem.text.split()) if title_elem is not None else ""

                summary_elem = entry.find("atom:summary", ns)
                abstract = " ".join(summary_elem.text.split()) if summary_elem is not None else ""

                pub_elem = entry.find("atom:published", ns)
                published_date = pub_elem.text.strip() if pub_elem is not None else ""

                categories = [
                    cat.attrib["term"]
                    for cat in entry.findall("atom:category", ns)
                    if "term" in cat.attrib
                ]

                # Ensure paper has an abstract and categories before adding
                if abstract and categories:
                    papers.append(
                        ArxivPaper(
                            paper_id=paper_id,
                            title=title,
                            abstract=abstract,
                            categories=categories,
                            published_date=published_date,
                        )
                    )

            print(f"Fetched {len(papers)} / {total_papers} papers...")

        except Exception as e:
            print(f"Error fetching batch at start={start_idx}: {e}")
            break

        # Respect ArXiv's API usage guidelines (at least 3 seconds between calls)
        if start_idx + batch_size < total_papers:
            time.sleep(3)

    print(f"Ingestion complete: retrieved {len(papers)} valid papers.")
    return papers
