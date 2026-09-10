"""
updater.py — Knowledge base updater for Task 3.
Orchestrates the ingestion pipeline → vector store update flow.
Preserves existing knowledge while adding new documents.
"""
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

from langchain_core.documents import Document
from src.knowledge.ingestion import ingest_text, ingest_file, ingest_url, get_ingestion_metadata
from src.retrieval.vector_store import append_to_index, index_exists
from src.core.config import KNOWLEDGE_INDEX

logger = logging.getLogger(__name__)


def update_from_text(text: str, name: str = "manual_entry", source_url: str = "") -> Dict:
    """
    Add plain text to the knowledge base.
    Returns status dict with counts and messages.
    """
    docs = ingest_text(text, name=name, source_url=source_url)
    return _commit_to_index(docs, source_name=name)


def update_from_file(file_path: str) -> Dict:
    """Add a local file (txt/pdf) to the knowledge base."""
    try:
        docs = ingest_file(file_path)
        return _commit_to_index(docs, source_name=Path(file_path).name)
    except Exception as e:
        logger.error(f"Error ingesting file {file_path}: {e}")
        return {"success": False, "message": str(e), "chunks_added": 0}


def update_from_url(url: str, name: Optional[str] = None) -> Dict:
    """Add a URL's content to the knowledge base."""
    try:
        docs = ingest_url(url, name=name)
        return _commit_to_index(docs, source_name=name or url)
    except Exception as e:
        logger.error(f"Error ingesting URL {url}: {e}")
        return {"success": False, "message": str(e), "chunks_added": 0}


def update_from_uploaded_bytes(
    file_bytes: bytes,
    filename: str,
) -> Dict:
    """
    Handle a file uploaded via Streamlit st.file_uploader.
    Writes to a temp path then ingests.
    """
    import tempfile, os

    ext = Path(filename).suffix.lower()
    if ext not in {".txt", ".md", ".pdf"}:
        return {
            "success": False,
            "message": f"Unsupported file type '{ext}'. Supported: .txt, .md, .pdf",
            "chunks_added": 0,
        }

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        result = update_from_file(tmp_path)
        result["filename"] = filename
        os.unlink(tmp_path)
        return result
    except Exception as e:
        return {"success": False, "message": str(e), "chunks_added": 0}


def _commit_to_index(docs: List[Document], source_name: str) -> Dict:
    """Append documents to the FAISS knowledge index."""
    if not docs:
        return {
            "success": True,
            "message": f"No new content to add for '{source_name}' (may be duplicate).",
            "chunks_added": 0,
        }
    try:
        append_to_index(docs, KNOWLEDGE_INDEX)
        return {
            "success": True,
            "message": f"Successfully added {len(docs)} chunks from '{source_name}' to the knowledge base.",
            "chunks_added": len(docs),
        }
    except Exception as e:
        logger.error(f"Failed to update index: {e}")
        return {
            "success": False,
            "message": f"Failed to update knowledge base: {e}",
            "chunks_added": 0,
        }


def get_knowledge_base_stats() -> Dict:
    """Return statistics about the current knowledge base."""
    docs_meta = get_ingestion_metadata()
    total_chunks = sum(d.get("chunks", 0) for d in docs_meta)
    return {
        "total_sources": len(docs_meta),
        "total_chunks": total_chunks,
        "index_exists": index_exists(KNOWLEDGE_INDEX),
        "sources": docs_meta,
    }


def answer_from_knowledge_base(question: str, history: str = "") -> Dict:
    """
    Query the dynamic knowledge base using RAG.
    Returns answer dict similar to other retrieval functions.
    """
    from src.retrieval.retriever import get_retriever
    from src.core.chatbot import build_llm
    from langchain_classic.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate

    if not index_exists(KNOWLEDGE_INDEX):
        return {
            "result": "The knowledge base is empty. Add documents first using the Knowledge Base Management panel.",
            "sources": [],
        }

    PROMPT_TEMPLATE = """You are a helpful assistant with access to a custom knowledge base.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information in the knowledge base."
Do NOT make up answers.

CONTEXT:
{context}

HISTORY:
{history}

QUESTION: {question}

ANSWER:"""

    try:
        llm = build_llm()
        retriever = get_retriever(KNOWLEDGE_INDEX, k=4)

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        ).partial(history=history)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
        )

        response = chain.invoke({"query": question, "history": history})
        sources = list({
            doc.metadata.get("source", "Unknown")
            for doc in response.get("source_documents", [])
        })
        return {"result": response.get("result", "No answer found."), "sources": sources}

    except Exception as e:
        logger.error(f"Knowledge base query error: {e}")
        return {"result": "Error querying knowledge base. Please try again.", "sources": []}
