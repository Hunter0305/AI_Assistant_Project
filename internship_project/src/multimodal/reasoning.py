"""
reasoning.py — Multimodal orchestration layer for Task 5.
Intelligently decides whether a task requires:
  - Text-only reasoning
  - Image analysis
  - Retrieval augmentation
  - Combined image + retrieval
  - Clarification

Maintains conversational context across image + text turns.
"""
import logging
from typing import Dict, Any, Optional

from src.core.chatbot import build_llm
from src.multimodal.image_processor import analyze_image, get_mime_type
from src.multimodal.validator import validate_response

logger = logging.getLogger(__name__)

# ── Intent detection keywords ─────────────────────────────────────────────────
IMAGE_REQUIRED_PHRASES = [
    "in this image", "in the image", "what does this show", "what is this",
    "analyze this", "describe this", "look at", "the chart", "the graph",
    "the diagram", "the picture", "what trend", "what does it show",
    "tell me about the image", "the figure",
]

AMBIGUOUS_PHRASES = [
    "this", "it", "that", "the thing", "what you see",
]


def _detect_intent(
    user_text: str,
    has_image: bool,
    has_image_context: bool,
) -> str:
    """
    Classify user intent to determine processing path.

    Returns one of:
        'image_analysis'     - analyze the current image
        'image_followup'     - follow-up on previously analyzed image
        'text_only'          - pure text Q&A
        'needs_clarification' - ambiguous request
    """
    text_lower = user_text.lower().strip()

    if has_image:
        return "image_analysis"

    if has_image_context:
        # Check if asking about previous image
        if any(phrase in text_lower for phrase in IMAGE_REQUIRED_PHRASES + AMBIGUOUS_PHRASES):
            return "image_followup"

    if not user_text.strip():
        return "needs_clarification"

    return "text_only"


def _handle_image_analysis(
    image_bytes: bytes,
    filename: str,
    user_text: str,
) -> Dict[str, Any]:
    """Process a newly uploaded image with optional user question."""
    mime = get_mime_type(filename)
    result = analyze_image(image_bytes, user_prompt=user_text, mime_type=mime)

    if not result["success"]:
        return {
            "response": f"I couldn't analyze the image: {result.get('error', 'Unknown error')}",
            "image_description": "",
            "processing_path": "image_analysis_failed",
            "valid": False,
        }

    description = result["description"]
    answer = result.get("answer", "")

    if answer and user_text.strip():
        response = f"**About your question:** {answer}\n\n**Image Summary:** {description}"
    else:
        response = f"**Image Analysis:**\n\n{description}"

    validated = validate_response(response, user_text or "Describe the image", description)

    return {
        "response": response,
        "image_description": description,
        "processing_path": "image_analysis",
        "valid": validated["is_valid"],
        "validation_notes": validated.get("notes", ""),
    }


def _handle_image_followup(
    user_text: str,
    image_context: str,
    conversation_history: str,
) -> Dict[str, Any]:
    """Handle follow-up questions about a previously analyzed image."""
    try:
        llm = build_llm()
        from langchain_core.messages import HumanMessage

        prompt = (
            f"You are analyzing an image that was previously described. "
            f"Use the image description below to answer the follow-up question.\n\n"
            f"PREVIOUS IMAGE DESCRIPTION:\n{image_context}\n\n"
            f"CONVERSATION HISTORY:\n{conversation_history}\n\n"
            f"FOLLOW-UP QUESTION: {user_text}\n\n"
            f"Answer based on what the image shows. If the question refers to 'it', "
            f"'this', 'that', etc., use the image context to resolve the reference.\n\n"
            f"ANSWER:"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content

        validated = validate_response(answer, user_text, image_context)
        return {
            "response": answer,
            "image_description": image_context,
            "processing_path": "image_followup",
            "valid": validated["is_valid"],
        }
    except Exception as e:
        logger.error(f"Image followup error: {e}")
        return {
            "response": "I had trouble processing your follow-up. Please try again.",
            "processing_path": "error",
            "valid": False,
        }


def _handle_text_only(user_text: str, history: str) -> Dict[str, Any]:
    """Handle text-only queries in multimodal mode."""
    try:
        llm = build_llm()
        from langchain_core.messages import HumanMessage

        prompt = (
            f"You are a helpful multimodal AI assistant. "
            f"Answer the user's question clearly and helpfully.\n\n"
            f"CONVERSATION HISTORY:\n{history}\n\n"
            f"USER: {user_text}\n\nASSISTANT:"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return {
            "response": response.content,
            "processing_path": "text_only",
            "valid": True,
        }
    except Exception as e:
        logger.error(f"Text reasoning error: {e}")
        return {
            "response": "I encountered an error. Please try again.",
            "processing_path": "error",
            "valid": False,
        }


def process_multimodal_request(
    user_text: str,
    image_bytes: Optional[bytes] = None,
    image_filename: str = "image.jpg",
    image_context: Optional[str] = None,
    conversation_history: str = "",
) -> Dict[str, Any]:
    """
    Main orchestration entry point for the Multimodal mode.

    Args:
        user_text: User's text input
        image_bytes: Raw bytes of uploaded image (None if no new image)
        image_filename: Original filename for MIME type detection
        image_context: Previously extracted image description (for follow-ups)
        conversation_history: Multi-turn conversation string

    Returns:
        dict with response, processing_path, image_description, valid
    """
    intent = _detect_intent(
        user_text=user_text,
        has_image=image_bytes is not None,
        has_image_context=bool(image_context),
    )

    if intent == "image_analysis":
        return _handle_image_analysis(image_bytes, image_filename, user_text)

    elif intent == "image_followup":
        return _handle_image_followup(user_text, image_context, conversation_history)

    elif intent == "needs_clarification":
        clarification = "Could you clarify what you'd like help with? You can upload an image or ask a question."
        return {
            "response": clarification,
            "processing_path": "clarification",
            "valid": True,
        }

    else:  # text_only
        return _handle_text_only(user_text, conversation_history)
