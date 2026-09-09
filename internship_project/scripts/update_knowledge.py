"""
update_knowledge.py — Manual knowledge base update script.
Ingests new documents from the knowledge_inbox directory or a specified path/URL.

Usage:
    cd internship_project

    # Update from knowledge inbox directory
    python scripts/update_knowledge.py

    # Update from specific file
    python scripts/update_knowledge.py --file path/to/document.pdf

    # Update from URL
    python scripts/update_knowledge.py --url https://example.com/article

    # Update from inline text
    python scripts/update_knowledge.py --text "New information to add" --name "my_doc"
"""
import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def update_from_inbox():
    """Scan and ingest all new files from the knowledge inbox."""
    inbox_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "knowledge_inbox"
    )
    os.makedirs(inbox_dir, exist_ok=True)

    from src.knowledge.scheduler import KnowledgeScheduler
    scheduler = KnowledgeScheduler(watch_dir=inbox_dir, interval_minutes=999)
    logs = scheduler.run_now()

    if logs:
        logger.info(f"Processed {len(logs)} files:")
        for entry in logs:
            status = "✅" if entry["success"] else "❌"
            logger.info(f"  {status} {entry['file']}: {entry['chunks_added']} chunks")
    else:
        logger.info("No new files found in knowledge inbox.")

    return len(logs)


def update_from_file(file_path: str):
    from src.knowledge.updater import update_from_file as _update
    logger.info(f"Ingesting file: {file_path}")
    result = _update(file_path)
    if result["success"]:
        logger.info(f"✅ {result['message']}")
    else:
        logger.error(f"❌ {result['message']}")
    return result["success"]


def update_from_url(url: str, name: str = None):
    from src.knowledge.updater import update_from_url as _update
    logger.info(f"Ingesting URL: {url}")
    result = _update(url, name=name)
    if result["success"]:
        logger.info(f"✅ {result['message']}")
    else:
        logger.error(f"❌ {result['message']}")
    return result["success"]


def update_from_text(text: str, name: str = "cli_text"):
    from src.knowledge.updater import update_from_text as _update
    logger.info(f"Ingesting text as '{name}'")
    result = _update(text, name=name)
    if result["success"]:
        logger.info(f"✅ {result['message']}")
    else:
        logger.error(f"❌ {result['message']}")
    return result["success"]


def show_stats():
    from src.knowledge.updater import get_knowledge_base_stats
    stats = get_knowledge_base_stats()
    logger.info("=" * 40)
    logger.info("Knowledge Base Statistics:")
    logger.info(f"  Total sources: {stats['total_sources']}")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Index active: {stats['index_exists']}")
    if stats["sources"]:
        logger.info("\n  Sources:")
        for src in stats["sources"]:
            logger.info(f"    - {src['name']} ({src['type']}, {src.get('chunks', 0)} chunks)")


def main():
    parser = argparse.ArgumentParser(description="Update the knowledge base")
    parser.add_argument("--file", help="Path to a file (.txt, .md, .pdf) to ingest")
    parser.add_argument("--url", help="URL to fetch and ingest")
    parser.add_argument("--text", help="Inline text to ingest")
    parser.add_argument("--name", help="Name/title for the document", default="cli_doc")
    parser.add_argument("--stats", action="store_true", help="Show knowledge base statistics")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.file:
        update_from_file(args.file)
    elif args.url:
        update_from_url(args.url, name=args.name)
    elif args.text:
        update_from_text(args.text, name=args.name)
    else:
        # Default: scan inbox
        logger.info("Scanning knowledge inbox for new files...")
        update_from_inbox()

    show_stats()


if __name__ == "__main__":
    main()
