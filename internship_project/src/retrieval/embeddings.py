"""
embeddings.py — Shared embedding loader using sentence-transformers.
Uses HuggingFaceEmbeddings (langchain_huggingface) wrapping all-MiniLM-L6-v2.
This replaces the deprecated HuggingFaceInstructEmbeddings from the training project.

Imports are lazy to avoid triggering the torch/transformers DLL chain at module load.
"""
import os
from functools import lru_cache
from src.core.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return a cached singleton HuggingFaceEmbeddings instance.
    The model is downloaded once and then reused across the entire app.
    Import is lazy to avoid module-level torch crash on Windows.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
