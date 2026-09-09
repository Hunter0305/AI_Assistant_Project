"""
test_sentiment.py — Tests for Task 1: Sentiment Analysis
Tests: positive, negative, neutral, ambiguous/mixed sentiment
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.sentiment.analyzer import analyze_sentiment, get_sentiment_emoji, get_sentiment_color


class TestSentimentAnalysis:
    """Task 1 Requirement: positive/negative/neutral detection."""

    def test_clearly_positive_message(self):
        """Clearly positive customer message should detect as positive."""
        label, score = analyze_sentiment("I love this course! It's amazing and very helpful!")
        assert label == "positive", f"Expected positive, got {label}"
        assert score > 0.5, f"Confidence should be > 0.5, got {score}"

    def test_clearly_negative_message(self):
        """Clearly negative customer complaint should detect as negative."""
        label, score = analyze_sentiment(
            "This is terrible! The support is awful and I'm very frustrated. Complete waste of money!"
        )
        assert label == "negative", f"Expected negative, got {label}"
        assert score > 0.5, f"Confidence should be > 0.5, got {score}"

    def test_neutral_question(self):
        """A neutral factual question should detect as neutral."""
        label, score = analyze_sentiment("How do I reset my password?")
        assert label == "neutral", f"Expected neutral, got {label}"

    def test_ambiguous_mixed_sentiment(self):
        """Ambiguous message — just verify it returns a valid label."""
        label, score = analyze_sentiment("It's okay I guess, not the best not the worst...")
        assert label in {"positive", "neutral", "negative"}, f"Invalid label: {label}"
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_empty_input(self):
        """Empty input should default to neutral."""
        label, score = analyze_sentiment("")
        assert label == "neutral"
        assert score == 0.5

    def test_returns_tuple(self):
        """Analyzer must return (str, float) tuple."""
        result = analyze_sentiment("Testing the analyzer")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_emoji_helper(self):
        """Emoji helper returns correct emoji per label."""
        assert get_sentiment_emoji("positive") == "😊"
        assert get_sentiment_emoji("negative") == "😟"
        assert get_sentiment_emoji("neutral") == "😐"

    def test_color_helper(self):
        """Color helper returns hex colors."""
        positive_color = get_sentiment_color("positive")
        assert positive_color.startswith("#")
        assert len(positive_color) == 7

    def test_all_valid_labels(self):
        """Analyzer always returns one of the three valid labels."""
        test_inputs = [
            "Hello",
            "I hate this!",
            "This is great!",
            "Can you help me?",
            "Refund please!!!",
        ]
        valid_labels = {"positive", "neutral", "negative"}
        for text in test_inputs:
            label, score = analyze_sentiment(text)
            assert label in valid_labels, f"'{text}' → invalid label: {label}"
