"""
test_multimodal.py — Tests for Task 5: Multimodal Assistant
Tests: intent detection, response validation, image processing logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.multimodal.validator import validate_response
from src.multimodal.image_processor import get_mime_type
from src.multimodal.reasoning import _detect_intent


class TestResponseValidator:
    """Task 5: response validation before display."""

    def test_valid_response_passes(self):
        """A proper answer should pass validation."""
        result = validate_response(
            "The image shows a bar chart with sales data trending upward.",
            "What trend is visible?",
            "Image description: bar chart showing sales data",
        )
        assert result["is_valid"] is True

    def test_empty_response_fails(self):
        """Empty response should fail validation."""
        result = validate_response("", "What do you see?", "Some context")
        assert result["is_valid"] is False

    def test_very_short_response_fails(self):
        """Very short response should fail."""
        result = validate_response("ok", "Describe the image in detail", "")
        assert result["is_valid"] is False

    def test_confidence_in_range(self):
        """Confidence should be 0.0 to 1.0."""
        result = validate_response("A reasonable answer to the question.", "Test question", "")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_returns_required_keys(self):
        """Result should have is_valid, confidence, notes."""
        result = validate_response("Some response", "Some question", "")
        assert "is_valid" in result
        assert "confidence" in result
        assert "notes" in result


class TestMimeTypeDetection:
    """Test image MIME type detection."""

    def test_jpg_mime(self):
        assert get_mime_type("photo.jpg") == "image/jpeg"

    def test_jpeg_mime(self):
        assert get_mime_type("image.JPEG") == "image/jpeg"

    def test_png_mime(self):
        assert get_mime_type("screenshot.png") == "image/png"

    def test_webp_mime(self):
        assert get_mime_type("image.webp") == "image/webp"

    def test_unknown_defaults_to_jpeg(self):
        assert get_mime_type("file.bmp") == "image/jpeg"


class TestIntentDetection:
    """Task 5: intelligent decision-making about processing path."""

    def test_image_upload_triggers_image_analysis(self):
        """New image upload should trigger image_analysis intent."""
        intent = _detect_intent("Tell me about this", has_image=True, has_image_context=False)
        assert intent == "image_analysis"

    def test_followup_without_image_uses_context(self):
        """Follow-up question when image context exists should use image_followup."""
        intent = _detect_intent("What is this thing?", has_image=False, has_image_context=True)
        assert intent == "image_followup"

    def test_text_only_without_image(self):
        """Regular text query without image context should be text_only."""
        intent = _detect_intent("How do I install Python?", has_image=False, has_image_context=False)
        assert intent == "text_only"

    def test_empty_text_triggers_clarification(self):
        """Empty text without image should ask for clarification."""
        intent = _detect_intent("", has_image=False, has_image_context=False)
        assert intent == "needs_clarification"
