"""
detector.py — Language detection for Task 6.
Uses langdetect (open-source, covers 55 languages) for automatic
language identification from user messages.

Supported languages for full response: en, hi, mr, es
All other detected languages receive an English response with a note.

Also handles mixed-language inputs (e.g., "मेरा password reset नहीं हो रहा").
"""
import logging
import re
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Supported language codes and their names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
}

# Fallback if langdetect is not available
_LANGDETECT_AVAILABLE = None


def _check_langdetect():
    global _LANGDETECT_AVAILABLE
    if _LANGDETECT_AVAILABLE is None:
        try:
            from langdetect import detect
            _LANGDETECT_AVAILABLE = True
        except ImportError:
            logger.warning("langdetect not installed. Run: pip install langdetect")
            _LANGDETECT_AVAILABLE = False
    return _LANGDETECT_AVAILABLE


def _detect_devanagari(text: str) -> Optional[str]:
    """
    Check for Devanagari script characters to distinguish Hindi vs Marathi.
    Returns 'hi' or 'mr' based on script presence and common markers.
    """
    devanagari_range = re.compile(r'[\u0900-\u097F]')
    if not devanagari_range.search(text):
        return None

    # Marathi-specific characters/words
    marathi_markers = ["आहे", "आणि", "मला", "कसे", "माझ्या", "तुम्ही", "नाही", "करा"]
    # Hindi-specific markers
    hindi_markers = ["है", "और", "मुझे", "कैसे", "मेरा", "आप", "नहीं", "करें"]

    text_lower = text.lower()
    mr_count = sum(1 for m in marathi_markers if m in text)
    hi_count = sum(1 for m in hindi_markers if m in text)

    if mr_count > hi_count:
        return "mr"
    return "hi"


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of input text.

    Returns:
        (language_code, confidence) — e.g., ("hi", 0.85)

    Handles:
        - Pure single-language text
        - Mixed-language text (e.g., English + Hindi)
        - Devanagari-script disambiguation (Hindi vs Marathi)
    """
    if not text or not text.strip():
        return "en", 1.0

    # First check Devanagari script for Hindi/Marathi disambiguation
    devanagari_lang = _detect_devanagari(text)
    if devanagari_lang:
        return devanagari_lang, 0.80

    # Try langdetect
    if _check_langdetect():
        try:
            from langdetect import detect, DetectorFactory
            from langdetect.lang_detect_exception import LangDetectException

            # Set seed for reproducibility
            DetectorFactory.seed = 42

            # For mixed text, try on the non-ASCII portion separately
            non_ascii = re.sub(r'[a-zA-Z0-9\s\W]', '', text)
            text_to_detect = non_ascii if len(non_ascii) > 10 else text

            lang = detect(text_to_detect)
            return lang, 0.80

        except Exception:
            pass

    # Fallback: check for Latin script (likely English or Spanish)
    if re.match(r'^[\w\s\W]+$', text, re.ASCII):
        return "en", 0.7

    return "en", 0.5


def get_language_name(lang_code: str) -> str:
    """Get human-readable language name from ISO code."""
    return SUPPORTED_LANGUAGES.get(lang_code, f"Language ({lang_code})")


def get_language_flag(lang_code: str) -> str:
    """Return a flag emoji for the language."""
    flags = {
        "en": "🇬🇧",
        "hi": "🇮🇳",
        "mr": "🇮🇳",
        "es": "🇪🇸",
    }
    return flags.get(lang_code, "🌐")


def is_supported_language(lang_code: str) -> bool:
    """Check if the language has full support (translation capability)."""
    return lang_code in SUPPORTED_LANGUAGES
