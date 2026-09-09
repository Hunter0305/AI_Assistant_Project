"""
build_indexes.py — One-time script to build all FAISS indexes.
Run this before starting the Streamlit app for the first time.

Usage:
    cd internship_project
    python scripts/build_indexes.py [--all] [--customer] [--medical] [--research]
"""
import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_customer_support():
    logger.info("=" * 50)
    logger.info("Building Customer Support Index...")
    from src.core.chatbot import build_customer_support_index
    success = build_customer_support_index()
    if success:
        logger.info("✅ Customer support index built successfully.")
    else:
        logger.error("❌ Failed to build customer support index.")
    return success


def build_medical():
    logger.info("=" * 50)
    logger.info("Building Medical Index (downloading MedQuAD)...")
    logger.info("This may take a few minutes on first run (fetching from GitHub).")
    from src.medical.ingestion import build_medical_index
    success = build_medical_index()
    if success:
        logger.info("✅ Medical index built successfully.")
    else:
        logger.error("❌ Failed to build medical index.")
    return success


def build_research():
    logger.info("=" * 50)
    logger.info("Building Research Index (fetching arXiv papers)...")
    logger.info("This may take a few minutes on first run.")
    from src.research.arxiv_loader import build_research_index
    success = build_research_index()
    if success:
        logger.info("✅ Research index built successfully.")
    else:
        logger.error("❌ Failed to build research index.")
    return success


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes for the AI Assistant")
    parser.add_argument("--all", action="store_true", help="Build all indexes")
    parser.add_argument("--customer", action="store_true", help="Build customer support index only")
    parser.add_argument("--medical", action="store_true", help="Build medical index only")
    parser.add_argument("--research", action="store_true", help="Build research index only")
    args = parser.parse_args()

    # Default to building customer support if no args
    if not any([args.all, args.customer, args.medical, args.research]):
        args.customer = True

    results = {}

    if args.all or args.customer:
        results["customer_support"] = build_customer_support()

    if args.all or args.medical:
        results["medical"] = build_medical()

    if args.all or args.research:
        results["research"] = build_research()

    logger.info("=" * 50)
    logger.info("Build Summary:")
    for name, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"  {status} {name}")

    if all(results.values()):
        logger.info("\n🎉 All indexes built! You can now run: streamlit run app.py")
    else:
        logger.warning("\n⚠️ Some indexes failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
