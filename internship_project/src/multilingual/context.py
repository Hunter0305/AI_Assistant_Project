"""
context.py — Multilingual conversation context manager for Task 6.
Maintains language continuity, entity preservation, and intent tracking
across language-switching turns in a conversation.

Requirement: "The assistant must preserve intent, context, entities,
and conversation state across language switches."
"""
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LanguageTurn:
    original_text: str
    english_text: str           # Normalized English version
    detected_language: str
    response_english: str       # English response
    response_localized: str     # Response in user's language
    entities: List[str] = field(default_factory=list)  # Extracted entities


class MultilingualContext:
    """
    Tracks conversation state across multilingual turns.
    Ensures context is preserved even when users switch languages.
    """

    def __init__(self):
        self.turns: List[LanguageTurn] = []
        self.current_language: str = "en"
        self.previous_language: str = "en"
        self.known_entities: List[str] = []  # Entities mentioned in conversation
        self.active_topic: Optional[str] = None  # Current conversation topic

    def add_turn(self, turn: LanguageTurn):
        """Record a completed conversation turn."""
        self.previous_language = self.current_language
        self.current_language = turn.detected_language
        self.turns.append(turn)

        # Update known entities
        for entity in turn.entities:
            if entity not in self.known_entities:
                self.known_entities.append(entity)

    def get_english_history(self, max_turns: int = 5) -> str:
        """
        Return conversation history in English regardless of original language.
        This is the key mechanism for cross-language context retention.
        """
        recent = self.turns[-max_turns:]
        lines = []
        for t in recent:
            lines.append(f"User: {t.english_text}")
            if t.response_english:
                lines.append(f"Assistant: {t.response_english[:300]}")
        return "\n".join(lines)

    def resolve_pronoun_reference(self, text: str, english_text: str) -> str:
        """
        If the query contains ambiguous pronouns and we have prior context,
        attempt to resolve them by appending known context.
        """
        pronouns = {"this", "it", "that", "they", "these", "those", "the above"}
        text_words = set(english_text.lower().split())

        if pronouns & text_words and self.turns:
            last_turn = self.turns[-1]
            # Append context hint
            context_hint = f"(referring to: {last_turn.active_topic or last_turn.english_text[:50]})"
            return f"{english_text} {context_hint}"
        return english_text

    def language_switched(self) -> bool:
        """Check if user switched language in the last turn."""
        return self.current_language != self.previous_language and len(self.turns) > 1

    def get_language_switch_note(self) -> str:
        """Generate a note about language switching for display."""
        if self.language_switched():
            from src.multilingual.detector import get_language_name
            prev = get_language_name(self.previous_language)
            curr = get_language_name(self.current_language)
            return f"🔄 Language switched: {prev} → {curr}"
        return ""

    def set_active_topic(self, topic: str):
        self.active_topic = topic

    def get_stats(self) -> Dict[str, Any]:
        languages_used = list({t.detected_language for t in self.turns})
        return {
            "total_turns": len(self.turns),
            "languages_used": languages_used,
            "current_language": self.current_language,
            "known_entities": self.known_entities[:10],
            "active_topic": self.active_topic,
        }


def process_multilingual_query(
    user_text: str,
    ml_context: MultilingualContext,
    chat_handler,  # Callable that takes (english_query, history) → str
) -> Dict[str, Any]:
    """
    Full multilingual processing pipeline.

    Steps:
    1. Detect language
    2. Normalize mixed-language input
    3. Translate to English
    4. Resolve pronoun references using context
    5. Process with underlying chat handler (English)
    6. Translate response back to user's language
    7. Update multilingual context

    Returns dict with:
        response: localized response
        detected_lang: str
        english_query: str
        language_switch_note: str
    """
    from src.multilingual.detector import detect_language, get_language_name
    from src.multilingual.processor import (
        translate_to_english,
        translate_from_english,
        normalize_mixed_language_query,
    )

    # Step 1: Detect language
    detected_lang, confidence = detect_language(user_text)
    logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")

    # Step 2: Normalize mixed-language input
    normalized = normalize_mixed_language_query(user_text, detected_lang)

    # Step 3: Translate to English
    english_query, was_translated = translate_to_english(normalized, detected_lang)

    # Step 4: Resolve pronoun references
    english_history = ml_context.get_english_history()
    english_query = ml_context.resolve_pronoun_reference(user_text, english_query)

    # Step 5: Process with underlying chat handler
    english_response = chat_handler(english_query, english_history)

    # Step 6: Translate response back
    if detected_lang != "en" and was_translated:
        localized_response = translate_from_english(english_response, detected_lang)
    else:
        localized_response = english_response

    # Step 7: Update context
    turn = LanguageTurn(
        original_text=user_text,
        english_text=english_query,
        detected_language=detected_lang,
        response_english=english_response,
        response_localized=localized_response,
    )
    ml_context.add_turn(turn)

    return {
        "response": localized_response,
        "detected_lang": detected_lang,
        "lang_name": get_language_name(detected_lang),
        "english_query": english_query,
        "was_translated": was_translated,
        "language_switch_note": ml_context.get_language_switch_note(),
        "confidence": confidence,
    }
