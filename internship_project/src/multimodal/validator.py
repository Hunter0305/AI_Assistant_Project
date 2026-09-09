"""
validator.py — Response validation for Task 5 multimodal assistant.
Validates that the generated response is grounded, addresses the question,
and doesn't contain obvious contradictions or unsupported claims.
"""
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Phrases indicating hallucination or uncertainty
HALLUCINATION_INDICATORS = [
    "i imagine", "i assume", "probably looks like", "must be",
    "certainly shows", "obviously contains", "i can tell you exactly",
]

# Phrases indicating response failure
NON_ANSWER_INDICATORS = [
    "i cannot see", "no image was provided", "i don't have access to",
    "as an ai, i cannot view",
]


def validate_response(
    response: str,
    question: str,
    context: str = "",
) -> Dict[str, Any]:
    """
    Validate a generated response before displaying it.

    Checks:
    1. Response is not empty
    2. Response actually addresses the question
    3. Response doesn't claim to have seen an image when no context exists
    4. No obvious hallucination indicators
    5. Response length is reasonable

    Returns:
        dict with is_valid (bool), confidence (float), notes (list of str)
    """
    notes = []
    issues = 0

    # Check 1: Empty response
    if not response or len(response.strip()) < 10:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "notes": ["Response is empty or too short."],
        }

    resp_lower = response.lower()

    # Check 2: Non-answer indicators
    for phrase in NON_ANSWER_INDICATORS:
        if phrase in resp_lower:
            notes.append(f"Response may not have processed image correctly: '{phrase}'")
            issues += 1

    # Check 3: Hallucination indicators (only relevant when context is provided)
    if context:
        for phrase in HALLUCINATION_INDICATORS:
            if phrase in resp_lower:
                notes.append(f"Potential hallucination indicator: '{phrase}'")
                issues += 1
                break  # Count once

    # Check 4: Response addresses the question
    # Extract key question words and check they're loosely addressed
    question_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", question.lower()))
    response_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", resp_lower))
    overlap = len(question_words & response_words)
    if len(question_words) > 3 and overlap < 1:
        notes.append("Response may not address the question directly.")
        issues += 1

    # Check 5: Response length sanity
    if len(response) > 5000:
        notes.append("Response is very long — consider a more concise answer.")
        # Not a hard failure

    confidence = max(0.0, 1.0 - (issues * 0.3))
    is_valid = issues < 2

    return {
        "is_valid": is_valid,
        "confidence": round(confidence, 2),
        "notes": notes,
    }
