"""
retriever.py — Unified retriever interface.
Wraps FAISS vector store to provide similarity search with configurable k and threshold.
All langchain_community imports are lazy to prevent torch DLL crash on Windows.
"""
from typing import List, Tuple, Optional
from src.retrieval.vector_store import load_index
from src.core.config import RETRIEVAL_K


def get_retriever(index_path: str, k: int = RETRIEVAL_K):
    """Return a LangChain retriever for an index, or None if index missing."""
    db = load_index(index_path)
    if db is None:
        return None
    return db.as_retriever(search_kwargs={"k": k})


def similarity_search(
    index_path: str,
    query: str,
    k: int = RETRIEVAL_K,
) -> List[Tuple]:
    """
    Perform similarity search and return (doc, score) tuples.
    Returns empty list if index does not exist.
    """
    db = load_index(index_path)
    if db is None:
        return []
    return db.similarity_search_with_relevance_scores(query, k=k)


def retrieve_documents(
    index_path: str,
    query: str,
    k: int = RETRIEVAL_K,
    min_score: float = 0.3,
) -> List:
    """
    Retrieve documents above a minimum relevance score threshold.
    Returns empty list when nothing relevant is found.
    """
    results = similarity_search(index_path, query, k=k)
    return [doc for doc, score in results if score >= min_score]
