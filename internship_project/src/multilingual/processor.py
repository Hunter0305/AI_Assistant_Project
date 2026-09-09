"""
processor.py — Cross-lingual text processing for Task 6.
Handles translation of queries to English for processing,
and translation of responses back to the user's detected language.
Uses deep-translator (open-source, free) as the translation backend.

Architecture:
    User message (any language)
         ↓ detect_language()
    Detected language code
         ↓ translate_to_english()
    English query
         ↓ [existing RAG chain — operates in English]
    English response
         ↓ translate_from_english()
    Response in user's language
"""
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Language code mapping for deep-translator
DEEPL_LANG_MAP = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "zh-cn": "chinese (simplified)",
    "ar": "arabic",
    "pt": "portuguese",
}

_translator_available = None


def _check_translator():
    global _translator_available
    if _translator_available is None:
        try:
            from deep_translator import GoogleTranslator
            _translator_available = True
        except ImportError:
            logger.warning("deep-translator not installed. Run: pip install deep-translator")
            _translator_available = False
    return _translator_available


def _deep_translate(text: str, source: str, target: str) -> str:
    """Translate text using deep-translator's GoogleTranslator."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=source, target=target)
        # deep-translator has a 5000 char limit per call; handle long text
        if len(text) <= 4500:
            return translator.translate(text)
        # For longer text, split and translate in chunks
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        translated_chunks = [translator.translate(c) for c in chunks]
        return " ".join(translated_chunks)
    except Exception as e:
        logger.error(f"Translation error ({source} → {target}): {e}")
        return text  # Return original on failure


def translate_to_english(text: str, source_lang: str) -> Tuple[str, bool]:
    """
    Translate text from source_lang to English.

    Returns:
        (translated_text, was_translated)
    """
    if source_lang == "en":
        return text, False

    if not _check_translator():
        logger.warning("Translator unavailable. Processing in original language.")
        return text, False

    # Map our language codes to deep-translator format
    src = DEEPL_LANG_MAP.get(source_lang, "auto")

    translated = _deep_translate(text, source=src, target="english")
    logger.debug(f"Translated from {source_lang}: '{text[:50]}' → '{translated[:50]}'")
    return translated, True


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate an English response to target_lang.
    Returns original text if translation fails or target is English.
    """
    if target_lang == "en":
        return text

    if not _check_translator():
        return text

    tgt = DEEPL_LANG_MAP.get(target_lang, "auto")
    if tgt == "auto":
        logger.warning(f"Unknown target language code: {target_lang}. Returning English.")
        return text

    return _deep_translate(text, source="english", target=tgt)


def normalize_mixed_language_query(text: str, detected_lang: str) -> str:
    """
    Handle mixed-language inputs (e.g., "मेरा password reset नहीं हो रहा").
    Normalizes the query by keeping it as-is but ensuring the semantic
    meaning is preserved by translating the whole thing to English.

    This is the key function for the mixed-language handling requirement.
    """
    if detected_lang == "en":
        return text

    # Check if text contains significant English content (>30% ASCII words)
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    total_words = len(text.split())
    english_ratio = english_words / max(total_words, 1)

    if english_ratio > 0.6:
        # Mostly English — process as-is
        return text
    else:
        # Translate the whole mixed query to English
        translated, _ = translate_to_english(text, detected_lang)
        return translated
