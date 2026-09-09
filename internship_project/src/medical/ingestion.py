"""
ingestion.py — MedQuAD dataset ingestion for Task 2.
Downloads a reproducible subset of the MedQuAD XML dataset from GitHub,
parses Q&A pairs, and builds a FAISS index.

MedQuAD source: https://github.com/abachaa/MedQuAD
Selected categories: NIH-based XML files covering diseases, symptoms, treatments.
"""
import os
import re
import json
import logging
import urllib.request
from xml.etree import ElementTree as ET
from typing import List, Dict
from datetime import datetime

from langchain_core.documents import Document
from src.core.config import MEDICAL_DATA_DIR, MEDICAL_INDEX, MEDQUAD_MAX_RECORDS
from src.retrieval.vector_store import create_index, index_exists

logger = logging.getLogger(__name__)

# MedQuAD XML files hosted on GitHub (raw content URLs)
# These are the most clinically relevant collections
MEDQUAD_SOURCES = [
    ("https://raw.githubusercontent.com/abachaa/MedQuAD/master/1_CancerGov_QA/0000228.xml", "cancer"),
    ("https://raw.githubusercontent.com/abachaa/MedQuAD/master/2_GARD_QA/0000002.xml", "rare_diseases"),
    ("https://raw.githubusercontent.com/abachaa/MedQuAD/master/4_MedlinePlus_QA/0000001.xml", "medlineplus"),
]

# We'll fetch the directory listing and download multiple files
MEDQUAD_RAW_BASE = "https://raw.githubusercontent.com/abachaa/MedQuAD/master"
MEDQUAD_API_BASE = "https://api.github.com/repos/abachaa/MedQuAD/contents"

MEDQUAD_FOLDERS = [
    "1_CancerGov_QA",
    "2_GARD_QA",
    "4_MedlinePlus_QA",
    "6_NIDDK_QA",
    "7_NINDS_QA",
    "9_CDC_QA",
    "10_MPlus_ADAM_QA",
]


def _fetch_folder_file_list(folder: str) -> List[str]:
    """Fetch list of XML file names in a MedQuAD GitHub folder."""
    url = f"{MEDQUAD_API_BASE}/{folder}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MedQuAD-Fetcher/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read())
            return [item["name"] for item in items if item["name"].endswith(".xml")]
    except Exception as e:
        logger.warning(f"Could not fetch folder listing for {folder}: {e}")
        return []


def _fetch_xml_content(folder: str, filename: str) -> str | None:
    """Fetch raw XML content from GitHub."""
    url = f"{MEDQUAD_RAW_BASE}/{folder}/{filename}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MedQuAD-Fetcher/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Could not fetch {url}: {e}")
        return None


def _parse_xml_to_qa(xml_content: str, source_folder: str) -> List[Dict]:
    """Parse a MedQuAD XML file and extract Q&A pairs."""
    pairs = []
    try:
        root = ET.fromstring(xml_content)
        topic_focus = root.attrib.get("Focus", "")
        topic_type = root.attrib.get("Type", "")

        for qa_pair in root.findall(".//QAPair"):
            question_el = qa_pair.find("Question")
            answer_el = qa_pair.find("Answer")

            if question_el is None or answer_el is None:
                continue

            question = (question_el.text or "").strip()
            answer = (answer_el.text or "").strip()

            if not question or not answer or len(answer) < 20:
                continue

            # Clean up whitespace
            question = re.sub(r"\s+", " ", question)
            answer = re.sub(r"\s+", " ", answer)

            pairs.append({
                "question": question,
                "answer": answer,
                "topic": topic_focus,
                "type": topic_type,
                "source": source_folder,
            })
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")
    return pairs


def _save_local_cache(qa_pairs: List[Dict]):
    """Save extracted QA pairs to local JSON cache."""
    os.makedirs(MEDICAL_DATA_DIR, exist_ok=True)
    cache_file = os.path.join(MEDICAL_DATA_DIR, "medquad_cache.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(qa_pairs)} QA pairs to {cache_file}")


def _load_local_cache() -> List[Dict]:
    """Load QA pairs from local cache if available."""
    cache_file = os.path.join(MEDICAL_DATA_DIR, "medquad_cache.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def download_medquad(max_records: int = MEDQUAD_MAX_RECORDS, force: bool = False) -> List[Dict]:
    """
    Download and parse MedQuAD data.
    Uses local cache if available and force=False.
    Returns a list of QA dicts.
    """
    if not force:
        cached = _load_local_cache()
        if cached:
            logger.info(f"Using cached MedQuAD data: {len(cached)} records")
            return cached[:max_records]

    logger.info("Downloading MedQuAD dataset from GitHub...")
    all_pairs = []

    for folder in MEDQUAD_FOLDERS:
        if len(all_pairs) >= max_records:
            break

        file_names = _fetch_folder_file_list(folder)
        # Limit files per folder for balance
        file_names = file_names[:25]

        for fname in file_names:
            if len(all_pairs) >= max_records:
                break
            xml = _fetch_xml_content(folder, fname)
            if xml:
                pairs = _parse_xml_to_qa(xml, folder)
                all_pairs.extend(pairs)

    logger.info(f"Downloaded {len(all_pairs)} MedQuAD QA pairs.")
    _save_local_cache(all_pairs)
    return all_pairs[:max_records]


def qa_pairs_to_documents(qa_pairs: List[Dict]) -> List[Document]:
    """Convert MedQuAD QA pairs to LangChain Document objects."""
    documents = []
    for pair in qa_pairs:
        content = (
            f"Question: {pair['question']}\n"
            f"Answer: {pair['answer']}"
        )
        metadata = {
            "source": pair.get("source", "MedQuAD"),
            "topic": pair.get("topic", ""),
            "type": pair.get("type", ""),
            "question": pair["question"],
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_medical_index(force: bool = False) -> bool:
    """
    Build the medical FAISS index from MedQuAD data.
    Returns True on success, False on failure.
    """
    if index_exists(MEDICAL_INDEX) and not force:
        logger.info("Medical index already exists. Skipping build.")
        return True

    try:
        qa_pairs = download_medquad()
        if not qa_pairs:
            logger.error("No MedQuAD data available.")
            return False

        documents = qa_pairs_to_documents(qa_pairs)
        create_index(documents, MEDICAL_INDEX)
        logger.info(f"Medical index built with {len(documents)} documents.")
        return True
    except Exception as e:
        logger.error(f"Failed to build medical index: {e}")
        return False
