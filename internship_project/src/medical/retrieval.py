"""
retrieval.py — Medical Q&A RAG chain for Task 2.
Retrieves relevant MedQuAD Q&A pairs and generates answers using Gemini.
Includes a medical safety layer: the assistant never claims to be a doctor
and always recommends professional consultation for diagnosis/treatment decisions.
"""
import logging
from typing import Dict, Any, List

from src.core.config import MEDICAL_INDEX
from src.retrieval.vector_store import index_exists
from src.medical.entities import extract_medical_entities

logger = logging.getLogger(__name__)

# ── Medical safety disclaimer ──────────────────────────────────────────────────
MEDICAL_DISCLAIMER = (
    "⚠️ **Medical Disclaimer**: This information is for educational purposes only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional for medical concerns."
)

# ── Dangerous question patterns ────────────────────────────────────────────────
DIAGNOSIS_PATTERNS = [
    "do i have", "am i sick", "is this cancer", "diagnose me",
    "what disease do i have", "tell me if i have", "am i dying",
    "is it serious", "should i be worried",
]

MEDICAL_PROMPT_TEMPLATE = """You are a knowledgeable medical information assistant. Your role is to provide
factual health information based ONLY on the medical knowledge base provided below.

IMPORTANT RULES:
1. You are NOT a doctor. Never diagnose, prescribe, or provide specific medical advice.
2. Always recommend consulting a healthcare professional for personal medical decisions.
3. Base your answer ONLY on the context provided. If context is insufficient, say so clearly.
4. Do NOT invent medical facts, statistics, or treatments not in the context.
5. For emergency symptoms (chest pain, difficulty breathing, severe bleeding), always advise calling emergency services.

MEDICAL KNOWLEDGE BASE:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Provide a clear, factual answer based on the knowledge base. If the information is not in the knowledge base,
say "I don't have sufficient information about this specific question in my knowledge base."
Always end with a recommendation to consult a healthcare professional when appropriate.

ANSWER:"""


def _is_dangerous_query(question: str) -> bool:
    """Check if a question is seeking personal diagnosis."""
    q_lower = question.lower()
    return any(pattern in q_lower for pattern in DIAGNOSIS_PATTERNS)


def _is_emergency_query(question: str) -> bool:
    """Check if query mentions emergency symptoms."""
    emergency_terms = [
        "chest pain", "can't breathe", "cannot breathe", "heart attack",
        "stroke", "unconscious", "not breathing", "overdose", "severe bleeding",
        "suicide", "self harm",
    ]
    q_lower = question.lower()
    return any(term in q_lower for term in emergency_terms)


def answer_medical_query(
    question: str,
    history: str = "",
) -> Dict[str, Any]:
    """
    Answer a medical question using MedQuAD RAG.

    Returns dict with:
        result: str — the answer
        sources: List[str] — source document identifiers
        disclaimer: str — always present safety disclaimer
        entities: Dict — extracted medical entities from the question
        is_emergency: bool
        insufficient_info: bool
    """
    result = {
        "result": "",
        "sources": [],
        "disclaimer": MEDICAL_DISCLAIMER,
        "entities": {},
        "is_emergency": False,
        "insufficient_info": False,
    }

    # Emergency check
    if _is_emergency_query(question):
        result["is_emergency"] = True
        result["result"] = (
            "🚨 **This sounds like a medical emergency.** "
            "Please call emergency services (911 / 112) immediately or go to the nearest emergency room. "
            "Do not delay seeking emergency medical care."
        )
        return result

    # Extract medical entities from question
    result["entities"] = extract_medical_entities(question)

    if not index_exists(MEDICAL_INDEX):
        result["result"] = (
            "The medical knowledge base hasn't been built yet. "
            "Please click 'Build Medical Index' in the sidebar."
        )
        result["insufficient_info"] = True
        return result

    # Check for personal diagnosis request
    if _is_dangerous_query(question):
        result["result"] = (
            "I'm not able to diagnose medical conditions — only a qualified healthcare provider can do that. "
            "I can provide general information about medical topics. "
            "Please consult a doctor for personal medical evaluation."
        )
        return result

    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_classic.chains import RetrievalQA
        from src.core.chatbot import build_llm
        from src.retrieval.retriever import get_retriever
        llm = build_llm()
        retriever = get_retriever(MEDICAL_INDEX, k=4)

        prompt = PromptTemplate(
            template=MEDICAL_PROMPT_TEMPLATE,
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

        # Check if answer indicates insufficient information
        insufficient_phrases = [
            "don't have sufficient", "not in my knowledge base",
            "no information", "I don't know", "cannot find",
        ]
        if any(p.lower() in answer.lower() for p in insufficient_phrases):
            result["insufficient_info"] = True

        result["result"] = answer

        # Collect source info
        sources = []
        for doc in response.get("source_documents", []):
            q = doc.metadata.get("question", "")
            topic = doc.metadata.get("topic", "")
            if q and len(q) > 10:
                sources.append(f"Q: {q[:80]}..." if len(q) > 80 else f"Q: {q}")
            elif topic:
                sources.append(f"Topic: {topic}")
        result["sources"] = list(dict.fromkeys(sources))[:3]  # deduplicate, max 3

        return result

    except Exception as e:
        logger.error(f"Medical query error: {e}")
        result["result"] = "I encountered an error processing your medical question. Please try again."
        result["error"] = str(e)
        return result
