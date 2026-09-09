"""
config.py — Central configuration for the AI Customer Service Assistant.
Loads environment variables from .env and exposes them as typed constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Google Gemini ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Hugging Face (optional, for gated models) ─────────────────────────────────
HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── LLM settings ──────────────────────────────────────────────────────────────
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ── FAISS index paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_indexes")
CUSTOMER_SUPPORT_INDEX = os.path.join(FAISS_DIR, "customer_support")
MEDICAL_INDEX = os.path.join(FAISS_DIR, "medical")
RESEARCH_INDEX = os.path.join(FAISS_DIR, "research")
KNOWLEDGE_INDEX = os.path.join(FAISS_DIR, "knowledge")

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
CUSTOMER_SUPPORT_CSV = os.path.join(DATA_DIR, "customer_support", "dataset.csv")
MEDICAL_DATA_DIR = os.path.join(DATA_DIR, "medical")
RESEARCH_DATA_DIR = os.path.join(DATA_DIR, "research")

# ── Retrieval settings ────────────────────────────────────────────────────────
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))

# ── MedQuAD settings ──────────────────────────────────────────────────────────
MEDQUAD_MAX_RECORDS: int = int(os.getenv("MEDQUAD_MAX_RECORDS", "800"))

# ── arXiv settings ────────────────────────────────────────────────────────────
ARXIV_MAX_PAPERS: int = int(os.getenv("ARXIV_MAX_PAPERS", "500"))
ARXIV_CATEGORY: str = os.getenv("ARXIV_CATEGORY", "cs.AI")

# ── Multilingual ──────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
}

# ── Knowledge base metadata store ─────────────────────────────────────────────
KNOWLEDGE_METADATA_FILE = os.path.join(FAISS_DIR, "knowledge_metadata.json")

def validate_config() -> dict:
    """Return a dict of config validation issues."""
    issues = {}
    if not GOOGLE_API_KEY:
        issues["GOOGLE_API_KEY"] = "Missing — set in .env file"
    return issues
