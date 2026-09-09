"""
summarizer.py — Research paper summarization for Task 4.
Generates structured summaries of individual papers using Gemini.
"""
import logging
from typing import Dict

from src.core.chatbot import build_llm

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Provide a structured summary of this research paper for a technical audience.

Paper Information:
Title: {title}
Authors: {authors}
Published: {published}
Categories: {categories}
Abstract: {abstract}

Generate a structured summary with these sections:
1. **Core Contribution**: What is the main contribution or finding?
2. **Problem Addressed**: What problem does this paper solve?
3. **Methodology**: What approach or method is used?
4. **Key Results**: What are the key results or claims?
5. **Significance**: Why is this work important to the field?
6. **Limitations**: Any noted limitations or future work?

Keep each section concise (2-3 sentences). Be factual and based only on the abstract provided."""


def summarize_paper(paper: Dict) -> str:
    """
    Generate a structured summary for a single paper.
    Returns the summary string.
    """
    try:
        llm = build_llm()
        from langchain_core.messages import HumanMessage

        prompt_text = SUMMARIZE_PROMPT.format(
            title=paper.get("title", "Unknown"),
            authors=paper.get("authors", "Unknown"),
            published=paper.get("published", "Unknown"),
            categories=paper.get("categories", "Unknown"),
            abstract=paper.get("abstract", paper.get("page_content", "No abstract available"))[:2000],
        )

        response = llm.invoke([HumanMessage(content=prompt_text)])
        return response.content

    except Exception as e:
        logger.error(f"Paper summarization error: {e}")
        return f"Could not generate summary: {e}"


def extract_key_concepts(abstract: str) -> str:
    """Extract key concepts from a paper abstract."""
    try:
        llm = build_llm()
        from langchain_core.messages import HumanMessage

        prompt = (
            f"Extract the 5-7 most important technical concepts/terms from this abstract. "
            f"Return them as a comma-separated list, ordered by importance.\n\nAbstract:\n{abstract[:1500]}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Concept extraction error: {e}")
        return "Could not extract concepts."
