"""
image_processor.py — Image analysis using Google Gemini Vision for Task 5.
Analyzes uploaded images and extracts structured descriptions.
Supports PNG, JPG/JPEG, WEBP formats.
"""
import base64
import logging
from typing import Optional, Dict, Any
from io import BytesIO

from src.core.config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

# Use gemini-1.5-flash which has vision capabilities
VISION_MODEL = "gemini-1.5-flash"


def _configure_gemini():
    import google.generativeai as genai
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set. Cannot use image analysis.")
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai


def analyze_image(
    image_bytes: bytes,
    user_prompt: str = "",
    mime_type: str = "image/jpeg",
) -> Dict[str, Any]:
    """
    Analyze an image using Gemini Vision.

    Args:
        image_bytes: Raw image bytes
        user_prompt: Optional user question about the image
        mime_type: MIME type of the image

    Returns:
        dict with:
            description: str — general image description
            answer: str — response to user_prompt (if provided)
            success: bool
            error: str (if failed)
    """
    try:
        genai = _configure_gemini()
        model = genai.GenerativeModel(VISION_MODEL)

        # Build content parts
        image_part = {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
        }

        # Step 1: Get a comprehensive image description
        description_prompt = (
            "Analyze this image comprehensively. Describe:\n"
            "1. What is shown in the image (main subject/content)\n"
            "2. Any text visible in the image\n"
            "3. Charts, graphs, or data visualizations (if present — describe the data)\n"
            "4. Colors, layout, and visual structure\n"
            "5. Any diagrams, figures, or technical content\n"
            "Be specific and detailed."
        )

        desc_response = model.generate_content([description_prompt, image_part])
        description = desc_response.text if desc_response.text else "Could not analyze image."

        # Step 2: If user provided a specific question, answer it
        answer = ""
        if user_prompt and user_prompt.strip():
            qa_prompt = (
                f"Based on this image, answer the following question:\n"
                f"Question: {user_prompt}\n\n"
                f"Be specific and reference what you actually see in the image."
            )
            qa_response = model.generate_content([qa_prompt, image_part])
            answer = qa_response.text if qa_response.text else "Could not answer based on image."

        return {
            "description": description,
            "answer": answer,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return {
            "description": "",
            "answer": "",
            "success": False,
            "error": str(e),
        }


def get_mime_type(filename: str) -> str:
    """Determine MIME type from filename."""
    ext = filename.lower().rsplit(".", 1)[-1]
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "image/jpeg")
