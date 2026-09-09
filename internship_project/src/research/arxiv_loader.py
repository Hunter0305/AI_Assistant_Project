"""
arxiv_loader.py — arXiv dataset loader for Task 4.
Uses the 'arxiv' Python package (official arXiv API) to reproducibly fetch
a Computer Science subset of papers. No Kaggle account needed.

Selected domain: Computer Science (cs.AI, cs.LG, cs.CL, cs.IR)
Fetches and caches paper metadata + abstracts locally.
Builds a FAISS index for semantic paper retrieval.
"""
import os
import json
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime

from langchain_core.documents import Document
from src.core.config import RESEARCH_DATA_DIR, RESEARCH_INDEX, ARXIV_MAX_PAPERS, ARXIV_CATEGORY

logger = logging.getLogger(__name__)

# CS sub-categories to fetch papers from
CS_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.IR", "cs.CV"]

CACHE_FILE = os.path.join(RESEARCH_DATA_DIR, "arxiv_papers.json")


def _fetch_arxiv_papers(
    categories: List[str],
    max_papers: int,
) -> List[Dict]:
    """
    Fetch papers from arXiv API using the 'arxiv' Python package.
    Returns a list of paper metadata dicts.
    """
    try:
        import arxiv
    except ImportError:
        raise ImportError("arxiv package required. Install with: pip install arxiv")

    papers = []
    per_category = max(1, max_papers // len(categories))

    client = arxiv.Client()

    for category in categories:
        if len(papers) >= max_papers:
            break

        logger.info(f"Fetching arXiv papers for category: {category}")
        try:
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            for result in client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [str(a) for a in result.authors[:5]],
                    "abstract": result.summary,
                    "categories": result.categories,
                    "published": result.published.isoformat() if result.published else "",
                    "arxiv_id": result.entry_id,
                    "pdf_url": result.pdf_url or "",
                    "primary_category": category,
                })
                if len(papers) >= max_papers:
                    break

            # Be polite to arXiv API
            time.sleep(1)

        except Exception as e:
            logger.warning(f"Error fetching {category}: {e}")
            continue

    return papers


def _save_cache(papers: List[Dict]):
    os.makedirs(RESEARCH_DATA_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved {len(papers)} papers to cache.")


def _load_cache() -> List[Dict]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def papers_to_documents(papers: List[Dict]) -> List[Document]:
    """Convert arXiv paper dicts to LangChain Documents."""
    docs = []
    for paper in papers:
        authors_str = ", ".join(paper.get("authors", []))
        categories_str = ", ".join(paper.get("categories", []))
        content = (
            f"Title: {paper['title']}\n"
            f"Authors: {authors_str}\n"
            f"Categories: {categories_str}\n"
            f"Published: {paper.get('published', 'Unknown')}\n"
            f"Abstract: {paper.get('abstract', '')}"
        )
        metadata = {
            "title": paper["title"],
            "authors": authors_str,
            "arxiv_id": paper.get("arxiv_id", ""),
            "pdf_url": paper.get("pdf_url", ""),
            "published": paper.get("published", ""),
            "primary_category": paper.get("primary_category", ""),
            "categories": categories_str,
            "source": "arXiv",
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def fetch_papers(force: bool = False, max_papers: int = ARXIV_MAX_PAPERS) -> List[Dict]:
    """
    Fetch arXiv CS papers. Uses cache if available and force=False.
    """
    if not force:
        cached = _load_cache()
        if cached:
            logger.info(f"Using cached arXiv data: {len(cached)} papers.")
            return cached[:max_papers]

    papers = _fetch_arxiv_papers(CS_CATEGORIES, max_papers)
    if papers:
        _save_cache(papers)
    return papers


def build_research_index(force: bool = False) -> bool:
    """
    Build the research FAISS index from arXiv papers.
    Returns True on success, False on failure.
    """
    from src.retrieval.vector_store import index_exists, create_index
    if index_exists(RESEARCH_INDEX) and not force:
        logger.info("Research index already exists. Skipping build.")
        return True

    try:
        papers = fetch_papers()
        if not papers:
            logger.error("No arXiv papers available.")
            return False

        documents = papers_to_documents(papers)
        create_index(documents, RESEARCH_INDEX)
        logger.info(f"Research index built with {len(documents)} papers.")
        return True
    except Exception as e:
        logger.error(f"Failed to build research index: {e}")
        return False


def get_cached_papers() -> List[Dict]:
    """Return all cached paper metadata."""
    return _load_cache()


def get_paper_categories_distribution() -> Dict[str, int]:
    """Return count of papers per category for visualization."""
    papers = _load_cache()
    counts: Dict[str, int] = {}
    for p in papers:
        cat = p.get("primary_category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return counts
