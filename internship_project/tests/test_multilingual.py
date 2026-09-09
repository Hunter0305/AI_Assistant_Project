"""
test_multilingual.py — Tests for Task 6: Multilingual Chat
Tests: language detection, translation flow, context preservation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.multilingual.detector import detect_language, get_language_name, get_language_flag, is_supported_language
from src.multilingual.context import MultilingualContext, LanguageTurn


class TestLanguageDetection:
    """Task 6: automatic language identification."""

    def test_english_detected(self):
        """English text should be detected as English."""
        lang, conf = detect_language("How do I reset my password?")
        assert lang == "en", f"Expected 'en', got '{lang}'"

    def test_hindi_devanagari_detected(self):
        """Hindi Devanagari script should be detected."""
        lang, conf = detect_language("मेरा पासवर्ड कैसे रीसेट करूं?")
        assert lang in {"hi", "mr"}, f"Expected 'hi' or 'mr', got '{lang}'"

    def test_spanish_detected(self):
        """Spanish text should be detected."""
        lang, conf = detect_language("¿Cómo restablezco mi contraseña?")
        assert lang == "es", f"Expected 'es', got '{lang}'"

    def test_mixed_language_returns_valid_code(self):
        """Mixed language input should return a valid language code."""
        lang, conf = detect_language("मेरा password reset नहीं हो रहा")
        assert isinstance(lang, str) and len(lang) >= 2

    def test_empty_string_defaults_to_english(self):
        """Empty string should default to English."""
        lang, conf = detect_language("")
        assert lang == "en"
        assert conf == 1.0

    def test_confidence_in_range(self):
        """Confidence should be between 0 and 1."""
        _, conf = detect_language("Hello world")
        assert 0.0 <= conf <= 1.0

    def test_get_language_name(self):
        """Should return human-readable language names."""
        assert get_language_name("en") == "English"
        assert get_language_name("hi") == "Hindi"
        assert get_language_name("es") == "Spanish"
        assert get_language_name("mr") == "Marathi"

    def test_get_language_flag(self):
        """Should return emoji flags."""
        flag = get_language_flag("en")
        assert flag  # Non-empty

    def test_supported_languages(self):
        """All required languages should be supported."""
        for code in ["en", "hi", "mr", "es"]:
            assert is_supported_language(code), f"Language '{code}' should be supported"


class TestMultilingualContext:
    """Task 6: context preservation across language switches."""

    def test_context_tracks_turns(self):
        """MultilingualContext should track turns."""
        ctx = MultilingualContext()
        turn = LanguageTurn(
            original_text="Hello",
            english_text="Hello",
            detected_language="en",
            response_english="Hi there!",
            response_localized="Hi there!",
        )
        ctx.add_turn(turn)
        assert ctx.get_stats()["total_turns"] == 1

    def test_language_switch_detection(self):
        """Should detect when user switches language."""
        ctx = MultilingualContext()
        turn1 = LanguageTurn("Hello", "Hello", "en", "Hi!", "Hi!")
        turn2 = LanguageTurn("Hola", "Hello", "es", "Hi!", "¡Hola!")
        ctx.add_turn(turn1)
        ctx.add_turn(turn2)
        assert ctx.language_switched() is True

    def test_english_history_preserved_across_languages(self):
        """English history should be accessible regardless of user language."""
        ctx = MultilingualContext()
        turn1 = LanguageTurn(
            "मेरा पासवर्ड",
            "My password",  # English version
            "hi",
            "Please try resetting it.",
            "कृपया इसे रीसेट करें।",
        )
        ctx.add_turn(turn1)
        history = ctx.get_english_history()
        assert "My password" in history

    def test_stats_return_correct_structure(self):
        """Stats dict should have all required keys."""
        ctx = MultilingualContext()
        stats = ctx.get_stats()
        required_keys = {"total_turns", "languages_used", "current_language", "known_entities", "active_topic"}
        assert required_keys.issubset(set(stats.keys()))


class TestProcessor:
    """Test translation processor (without actual API calls when possible)."""

    def test_english_passes_through_unchanged(self):
        """English text should pass through translate_to_english unchanged."""
        from src.multilingual.processor import translate_to_english
        text = "How do I reset my password?"
        result, was_translated = translate_to_english(text, "en")
        assert result == text
        assert was_translated is False

    def test_english_from_english_passthrough(self):
        """translate_from_english with target=en should return original."""
        from src.multilingual.processor import translate_from_english
        text = "This is the answer."
        result = translate_from_english(text, "en")
        assert result == text
