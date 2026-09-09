"""
ingestion.py — Multi-source ingestion pipeline for Task 3.
Supports: plain text, PDF files, URLs, and local documents.
All documents are tagged with metadata (source, type, timestamp).
Duplicate detection prevents re-ingesting the same content.
"""
import os
import re
import json
import hashlib
import logging
import urllib.request
from typing import List, Dict, Optional
from datetime import datetime

from langchain_core.documents import Document
from src.core.config import KNOWLEDGE_INDEX, KNOWLEDGE_METADATA_FILE

logger = logging.getLogger(__name__)

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple recursive text splitter (no external dependency)."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    for sep in separators:
        if sep and sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                test = current + (sep if current else "") + part
                if len(test) > chunk_size and current:
                    chunks.append(current.strip())
                    current = part
                else:
                    current = test
            if current.strip():
                chunks.append(current.strip())
            # Add overlap
            result = []
            for i, chunk in enumerate(chunks):
                if i > 0 and overlap > 0:
                    prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) >= overlap else chunks[i-1]
                    chunk = prev_end + " " + chunk
                result.append(chunk)
            return result
    # No separator found — hard split
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]



# ── Metadata store ────────────────────────────────────────────────────────────
def _load_metadata() -> Dict:
    if os.path.exists(KNOWLEDGE_METADATA_FILE):
        with open(KNOWLEDGE_METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "content_hashes": []}


def _save_metadata(meta: Dict):
    os.makedirs(os.path.dirname(KNOWLEDGE_METADATA_FILE), exist_ok=True)
    with open(KNOWLEDGE_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def _content_hash(text: str) -> str:
    """SHA-256 hash of content for duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _is_duplicate(content_hash: str, meta: Dict) -> bool:
    return content_hash in meta.get("content_hashes", [])


def _register_document(
    name: str,
    source_type: str,
    source_url: str,
    content_hash: str,
    num_chunks: int,
    meta: Dict,
):
    meta.setdefault("documents", []).append({
        "name": name,
        "type": source_type,
        "url": source_url,
        "hash": content_hash,
        "chunks": num_chunks,
        "ingested_at": datetime.now().isoformat(),
    })
    meta.setdefault("content_hashes", []).append(content_hash)
    _save_metadata(meta)


# ── Content readers ───────────────────────────────────────────────────────────
def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError("pypdf is required for PDF ingestion. Install with: pip install pypdf")
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {path}: {e}")


def _read_url(url: str) -> str:
    try:
        from urllib.request import urlopen, Request
        headers = {"User-Agent": "KnowledgeBaseBot/1.0"}
        req = Request(url, headers=headers)
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")

        # Strip HTML tags if it's HTML content
        if "<html" in raw[:500].lower():
            # Simple HTML tag stripping
            clean = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
            clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL | re.IGNORECASE)
            clean = re.sub(r"<[^>]+>", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()
            return clean[:50000]  # Limit to 50K chars
        return raw[:50000]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL {url}: {e}")


def _clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r" {3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ── Main ingestion functions ───────────────────────────────────────────────────
def ingest_text(
    text: str,
    name: str = "manual_entry",
    source_url: str = "",
) -> List[Document]:
    """
    Ingest plain text. Returns list of chunked Document objects.
    Skips if content is duplicate.
    """
    meta = _load_metadata()
    text = _clean_text(text)
    content_hash = _content_hash(text)

    if _is_duplicate(content_hash, meta):
        logger.info(f"Duplicate content detected for '{name}'. Skipping.")
        return []

    chunks = _split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source": name,
                "type": "text",
                "url": source_url,
                "chunk": i,
                "ingested_at": datetime.now().isoformat(),
            },
        ))

    _register_document(name, "text", source_url, content_hash, len(docs), meta)
    logger.info(f"Ingested text '{name}' as {len(docs)} chunks.")
    return docs


def ingest_file(file_path: str) -> List[Document]:
    """
    Ingest a local file (txt or pdf).
    Detects type from extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path)

    if ext == ".pdf":
        text = _read_pdf_file(file_path)
        source_type = "pdf"
    elif ext in {".txt", ".md", ".rst", ".csv"}:
        text = _read_text_file(file_path)
        source_type = "text"
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .md, .pdf")

    meta = _load_metadata()
    text = _clean_text(text)
    content_hash = _content_hash(text)

    if _is_duplicate(content_hash, meta):
        logger.info(f"Duplicate content for '{name}'. Skipping.")
        return []

    chunks = _split_text(text)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": name,
                "type": source_type,
                "url": file_path,
                "chunk": i,
                "ingested_at": datetime.now().isoformat(),
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    _register_document(name, source_type, file_path, content_hash, len(docs), meta)
    logger.info(f"Ingested file '{name}' as {len(docs)} chunks.")
    return docs


def ingest_url(url: str, name: Optional[str] = None) -> List[Document]:
    """Ingest content from a URL."""
    if name is None:
        name = url.split("//")[-1][:60]

    text = _read_url(url)
    return ingest_text(text, name=name, source_url=url)


def get_ingestion_metadata() -> List[Dict]:
    """Return list of all ingested document metadata records."""
    return _load_metadata().get("documents", [])
