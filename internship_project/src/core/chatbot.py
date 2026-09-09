"""
chatbot.py — Core Customer Support RAG chain.
Modernized version of the original training project's langchain_helper.py.
Original: GooglePalm + HuggingFaceInstructEmbeddings + langchain 0.0.339
Modernized: Gemini 1.5 Flash + sentence-transformers + langchain_community

Original functionality preserved:
- CSV FAQ dataset from Nullclass
- FAISS vector database
- RetrievalQA chain
- "I don't know" grounding (no hallucination)

New additions:
- Sentiment-aware prompt adaptation (Task 1)
- Multi-turn conversation context
- Friendly error handling

All LLM-related imports are LAZY to prevent torch/CUDA DLL crashes on Windows.
"""
import logging
from typing import Optional, Dict, Any

from src.core.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    CUSTOMER_SUPPORT_CSV,
    CUSTOMER_SUPPORT_INDEX,
)
from src.retrieval.vector_store import index_exists

logger = logging.getLogger(__name__)

# ── Sentiment-aware prompt templates ─────────────────────────────────────────
_BASE_CONTEXT = """You are a helpful customer service assistant for Nullclass, an e-learning company.
Answer the customer's question using ONLY the provided FAQ context.
If the answer is not found in the context, say "I don't have information about that — please contact our support team."
Do NOT make up answers or information not present in the context.

FAQ CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

CUSTOMER QUESTION: {question}"""

_POSITIVE_PREFIX = (
    "The customer seems happy. Keep the tone warm and enthusiastic.\n\n"
)
_NEGATIVE_PREFIX = (
    "The customer seems frustrated or upset. Be especially empathetic, "
    "acknowledge their frustration, and provide clear actionable help.\n\n"
)
_NEUTRAL_PREFIX = (
    "Provide a direct, professional, and informative response.\n\n"
)


def _get_sentiment_prefix(sentiment: str) -> str:
    mapping = {
        "positive": _POSITIVE_PREFIX,
        "negative": _NEGATIVE_PREFIX,
        "neutral": _NEUTRAL_PREFIX,
    }
    return mapping.get(sentiment, _NEUTRAL_PREFIX)


def build_llm():
    """Build and return the Gemini LLM. Lazy import to avoid torch cascade."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file."
        )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )


def build_customer_support_index() -> bool:
    """
    Build the customer support FAISS index from the training dataset CSV.
    Equivalent to create_vector_db() in the original langchain_helper.py.
    Returns True on success, False on failure.
    """
    try:
        from langchain_community.document_loaders import CSVLoader
        from src.retrieval.vector_store import create_index
        loader = CSVLoader(
            file_path=CUSTOMER_SUPPORT_CSV,
            source_column="prompt",
            encoding="latin1",
        )
        docs = loader.load()
        create_index(docs, CUSTOMER_SUPPORT_INDEX)
        logger.info(f"Customer support index built with {len(docs)} documents.")
        return True
    except Exception as e:
        logger.error(f"Failed to build customer support index: {e}")
        return False


def answer_customer_query(
    question: str,
    sentiment: str = "neutral",
    history: str = "",
) -> Dict[str, Any]:
    """
    Answer a customer support question using RAG.
    Equivalent to get_qa_chain()(question) in the original langchain_helper.py.
    Sentiment parameter modifies the system instruction tone.

    Returns dict with 'result' and 'sources'.
    """
    if not index_exists(CUSTOMER_SUPPORT_INDEX):
        return {
            "result": (
                "The knowledge base hasn't been built yet. "
                "Please click 'Build Knowledge Base' in the sidebar."
            ),
            "sources": [],
        }

    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_classic.chains import RetrievalQA
        from src.retrieval.retriever import get_retriever

        llm = build_llm()
        retriever = get_retriever(CUSTOMER_SUPPORT_INDEX)

        sentiment_prefix = _get_sentiment_prefix(sentiment)
        full_template = sentiment_prefix + _BASE_CONTEXT + "\n\nANSWER:"

        prompt = PromptTemplate(
            template=full_template,
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

        response = chain.invoke({
            "query": question,
            "history": history,
        })

        sources = []
        for doc in response.get("source_documents", []):
            src = doc.metadata.get("source", "")
            if src and src not in sources:
                sources.append(src)

        return {
            "result": response.get("result", "I don't know."),
            "sources": sources,
        }

    except Exception as e:
        logger.error(f"Error in answer_customer_query: {e}")
        return {
            "result": "I encountered an error while processing your question. Please try again.",
            "sources": [],
            "error": str(e),
        }
