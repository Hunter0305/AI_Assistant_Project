"""
analyzer.py — Sentiment Analysis for Task 1.
Uses a dual-model approach: VADER (rule-based, fast) + TextBlob (ML-based).
Both are open-source, require no GPU, and run entirely on CPU.

Model selection rationale:
- VADER: Specifically tuned for social/customer-service text. Provides
  compound score in [-1, 1]. Excellent for short conversational sentences.
- TextBlob: Provides polarity confirmation. Used to compute ensemble confidence.

Sentiment labels: positive, neutral, negative
Confidence: 0.0 – 1.0 derived from VADER compound score magnitude.

How sentiment changes responses (Task 1 requirement):
- The label and score are stored on the Message object (conversation.py)
- chatbot.py reads the label and selects a sentiment-adapted system prompt
  prefix that changes the LLM's tone instruction.
"""
import re
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid startup delay
_vader_analyzer = None
_vader_loaded = False


def _get_vader():
    global _vader_analyzer, _vader_loaded
    if not _vader_loaded:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed. Install with: pip install vaderSentiment")
            _vader_analyzer = None
        _vader_loaded = True
    return _vader_analyzer


def _textblob_sentiment(text: str) -> float:
    """Return TextBlob polarity in [-1.0, 1.0]. Returns 0.0 on failure."""
    try:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity
    except ImportError:
        logger.warning("textblob not installed. Install with: pip install textblob")
        return 0.0
    except Exception:
        return 0.0


def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Analyze the sentiment of input text using VADER + TextBlob ensemble.

    Returns:
        label (str): 'positive', 'neutral', or 'negative'
        confidence (float): 0.0–1.0

    Algorithm:
        1. Run VADER to get compound score.
        2. Run TextBlob to get polarity.
        3. Average the two signals (normalized to [-1,1]).
        4. Classify: avg > 0.1 → positive, avg < -0.1 → negative, else neutral.
        5. Confidence = |avg| mapped to [0.5, 1.0].

    Test cases:
        "I love this product!"        → positive (~0.85 confidence)
        "This is terrible service!"   → negative (~0.80 confidence)
        "How do I reset my password?" → neutral (~0.50 confidence)
        "It's okay I guess..."        → neutral/slightly positive (~0.52 confidence)
    """
    if not text or not text.strip():
        return "neutral", 0.5

    # Strip very short texts that may confuse models
    clean = re.sub(r"\s+", " ", text.strip())

    vader = _get_vader()
    vader_score = 0.0
    if vader is not None:
        scores = vader.polarity_scores(clean)
        vader_score = scores["compound"]  # -1 to 1

    tb_score = _textblob_sentiment(clean)  # -1 to 1

    # Ensemble: weighted average (VADER is generally better for short text)
    ensemble = (0.65 * vader_score) + (0.35 * tb_score)

    # Classification thresholds
    if ensemble > 0.1:
        label = "positive"
    elif ensemble < -0.1:
        label = "negative"
    else:
        label = "neutral"

    # Confidence: map |ensemble| from [0, 1] → [0.5, 1.0]
    confidence = round(0.5 + 0.5 * min(abs(ensemble), 1.0), 3)

    return label, confidence


def get_sentiment_emoji(label: str) -> str:
    """Return an emoji for display in the Streamlit UI."""
    return {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(label, "😐")


def get_sentiment_color(label: str) -> str:
    """Return a hex color for the sentiment indicator badge."""
    return {
        "positive": "#2ecc71",
        "neutral": "#95a5a6",
        "negative": "#e74c3c",
    }.get(label, "#95a5a6")
