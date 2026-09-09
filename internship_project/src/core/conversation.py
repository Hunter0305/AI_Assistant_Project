"""
conversation.py — Manages multi-turn conversation state including history,
detected sentiment, language, and active mode.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Message:
    role: str           # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sentiment: Optional[str] = None          # positive / neutral / negative
    sentiment_score: Optional[float] = None  # confidence
    language: Optional[str] = None           # detected ISO code
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationState:
    """
    Central conversation state object. One instance per Streamlit session.
    Tracks messages, language, sentiment history, and mode.
    """

    def __init__(self):
        self.messages: List[Message] = []
        self.mode: str = "customer_support"  # active tab/mode
        self.current_language: str = "en"
        self.last_sentiment: str = "neutral"
        self.last_sentiment_score: float = 0.0
        self.image_context: Optional[str] = None   # last analyzed image description
        self.uploaded_image_bytes: Optional[bytes] = None
        self.session_start: datetime = datetime.now()
        self.metadata: Dict[str, Any] = {}          # mode-specific extras

    def add_user_message(
        self,
        content: str,
        sentiment: str = "neutral",
        sentiment_score: float = 0.0,
        language: str = "en",
    ) -> Message:
        msg = Message(
            role="user",
            content=content,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            language=language,
        )
        self.messages.append(msg)
        self.last_sentiment = sentiment
        self.last_sentiment_score = sentiment_score
        self.current_language = language
        return msg

    def add_assistant_message(self, content: str, metadata: Dict[str, Any] | None = None) -> Message:
        msg = Message(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        return msg

    def get_history_text(self, max_turns: int = 6) -> str:
        """Return the last N turns formatted as a conversation string."""
        recent = self.messages[-max_turns * 2:]
        lines = []
        for m in recent:
            prefix = "User" if m.role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def get_last_user_message(self) -> Optional[str]:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return None

    def clear(self):
        self.messages.clear()
        self.image_context = None
        self.uploaded_image_bytes = None

    def set_image_context(self, description: str, image_bytes: bytes):
        self.image_context = description
        self.uploaded_image_bytes = image_bytes

    def to_streamlit_display(self) -> List[Dict]:
        """Convert messages to a simple list for st.chat_message display."""
        return [{"role": m.role, "content": m.content} for m in self.messages]
