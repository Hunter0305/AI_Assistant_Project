"""
retrieval.py — Research paper retrieval and Q&A for Task 4.
Retrieves relevant arXiv papers and uses Gemini to explain concepts,
summarize papers, and answer follow-up questions with context retention.
"""
import logging
from typing import Dict, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

from src.core.config import RESEARCH_INDEX
from src.core.chatbot import build_llm
from src.retrieval.vector_store import index_exists
from src.retrieval.retriever import get_retriever, retrieve_documents

logger = logging.getLogger(__name__)

RESEARCH_PROMPT_TEMPLATE = """You are an expert AI research assistant with deep knowledge of computer science and machine learning.
Your task is to explain scientific concepts clearly and accurately, citing the research papers provided.

RETRIEVED RESEARCH PAPERS:
{context}

CONVERSATION HISTORY (for context retention):
{history}

USER QUESTION: {question}

Instructions:
1. Use the retrieved papers as your primary evidence source.
2. Clearly cite paper titles when referencing specific work.
3. Explain complex concepts in accessible language.
4. Distinguish between: "According to retrieved papers:" and "General knowledge:".
5. If the papers don't contain enough information, say so explicitly.
6. For follow-up questions, maintain context from the conversation history.
7. Do NOT invent paper citations or authors.

RESPONSE (include: concept explanation, supporting evidence from papers, citations):"""


def search_papers(query: str, k: int = 5) -> List[Dict]:
    """
    Search for relevant papers and return their metadata.
    Used for the paper search UI component.
    """
    if not index_exists(RESEARCH_INDEX):
        return []

    docs = retrieve_documents(RESEARCH_INDEX, query, k=k, min_score=0.2)
    results = []
    for doc in docs:
        meta = doc.metadata
        results.append({
            "title": meta.get("title", "Unknown Title"),
            "authors": meta.get("authors", ""),
            "abstract": doc.page_content[:500],
            "arxiv_id": meta.get("arxiv_id", ""),
            "pdf_url": meta.get("pdf_url", ""),
            "published": meta.get("published", ""),
            "categories": meta.get("categories", ""),
            "primary_category": meta.get("primary_category", ""),
        })
    return results


def answer_research_query(
    question: str,
    history: str = "",
) -> Dict[str, Any]:
    """
    Answer a research question using arXiv RAG.
    Returns:
        result: str — generated explanation
        papers: List[Dict] — retrieved paper metadata
        evidence_available: bool
    """
    if not index_exists(RESEARCH_INDEX):
        return {
            "result": (
                "The research index hasn't been built yet. "
                "Please click 'Build Research Index' in the sidebar."
            ),
            "papers": [],
            "evidence_available": False,
        }

    try:
        llm = build_llm()
        retriever = get_retriever(RESEARCH_INDEX, k=5)

        prompt = PromptTemplate(
            template=RESEARCH_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        ).partial(history=history)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context",
            },
        )

        response = chain.invoke({"query": question, "history": history})
        answer = response.get("result", "")

        # Collect paper metadata from source documents
        retrieved_papers = []
        for doc in response.get("source_documents", []):
            meta = doc.metadata
            retrieved_papers.append({
                "title": meta.get("title", "Unknown"),
                "authors": meta.get("authors", ""),
                "arxiv_id": meta.get("arxiv_id", ""),
                "pdf_url": meta.get("pdf_url", ""),
                "published": meta.get("published", ""),
                "categories": meta.get("categories", ""),
            })

        # Check if evidence was actually used
        evidence_phrases = ["according to", "paper", "study", "research", "authors"]
        evidence_used = any(p in answer.lower() for p in evidence_phrases)

        return {
            "result": answer,
            "papers": retrieved_papers,
            "evidence_available": len(retrieved_papers) > 0,
            "evidence_used_in_answer": evidence_used,
        }

    except Exception as e:
        logger.error(f"Research query error: {e}")
        return {
            "result": "I encountered an error processing your research question. Please try again.",
            "papers": [],
            "evidence_available": False,
            "error": str(e),
        }
