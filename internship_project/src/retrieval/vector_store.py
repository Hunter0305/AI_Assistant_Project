"""
vector_store.py — FAISS-based vector store wrapper.
Supports creating, loading, and appending to named FAISS indexes.
Preserves existing data when adding new documents (no destructive rebuild).

FAISS and langchain_community imports are lazy to prevent torch DLL crash at module load.
"""
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _get_faiss_and_embeddings():
    """Lazy loader for FAISS and embeddings — avoids torch DLL crash at import time."""
    from langchain_community.vectorstores import FAISS
    from src.retrieval.embeddings import get_embeddings
    return FAISS, get_embeddings()


def create_index(documents: List, index_path: str):
    """
    Create a new FAISS index from documents and persist to disk.
    WARNING: This overwrites any existing index at index_path.
    """
    FAISS, embeddings = _get_faiss_and_embeddings()
    vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectordb.save_local(index_path)
    logger.info(f"Created index at {index_path} with {len(documents)} documents.")
    return vectordb


def load_index(index_path: str):
    """Load an existing FAISS index from disk. Returns None if not found."""
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        logger.warning(f"No index found at {index_path}")
        return None
    FAISS, embeddings = _get_faiss_and_embeddings()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def append_to_index(documents: List, index_path: str):
    """
    Append new documents to an existing FAISS index.
    If no index exists, creates a new one.
    This is the key function for Task 3 (dynamic knowledge base expansion).
    """
    FAISS, embeddings = _get_faiss_and_embeddings()
    existing = load_index(index_path)
    if existing is None:
        logger.info(f"No existing index at {index_path}. Creating new index.")
        return create_index(documents, index_path)

    new_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    existing.merge_from(new_store)
    existing.save_local(index_path)
    logger.info(f"Appended {len(documents)} documents to index at {index_path}.")
    return existing


def index_exists(index_path: str) -> bool:
    """Check whether a FAISS index exists at the given path. Pure filesystem check — no torch."""
    return os.path.exists(os.path.join(index_path, "index.faiss"))
