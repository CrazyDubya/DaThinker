"""
Conversation Context Manager

Tracks conversation state, history, and contextual signals
for the adaptive system to make informed decisions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import hashlib


class ConversationState(Enum):
    """Current state of the conversation."""
    INITIAL = "initial"
    ENGAGED = "engaged"
    SUSPICIOUS = "suspicious"
    HOSTILE = "hostile"
    CONFUSED = "confused"
    FLOWING = "flowing"
    TERMINATED = "terminated"


class TrustLevel(Enum):
    """Trust level for the current conversation."""
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    ZERO = 0


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Generate a hash of the message content."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class ConversationContext:
    """
    Manages the full context of a conversation including history,
    state tracking, and anomaly signals.
    """
    session_id: str
    history: List[Message] = field(default_factory=list)
    state: ConversationState = ConversationState.INITIAL
    trust_level: TrustLevel = TrustLevel.MEDIUM
    anomaly_score: float = 0.0
    manipulation_attempts: int = 0
    injection_attempts: int = 0
    logical_coherence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a new message to the conversation history."""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.history.append(msg)
        self._update_state()
        return msg

    def _update_state(self):
        """Update conversation state based on recent activity."""
        if self.anomaly_score > 0.7:
            self.state = ConversationState.HOSTILE
            self.trust_level = TrustLevel.ZERO
        elif self.anomaly_score > 0.4:
            self.state = ConversationState.SUSPICIOUS
            self.trust_level = TrustLevel.LOW
        elif self.logical_coherence_score < 0.3:
            self.state = ConversationState.CONFUSED
        elif len(self.history) > 2 and self.anomaly_score < 0.2:
            self.state = ConversationState.FLOWING
            if self.anomaly_score < 0.1:
                self.trust_level = TrustLevel.HIGH
        elif len(self.history) > 0:
            self.state = ConversationState.ENGAGED

    def get_recent_history(self, n: int = 5) -> List[Message]:
        """Get the most recent n messages."""
        return self.history[-n:] if len(self.history) >= n else self.history

    def record_manipulation_attempt(self):
        """Record a detected manipulation attempt."""
        self.manipulation_attempts += 1
        self.anomaly_score = min(1.0, self.anomaly_score + 0.2)
        self._update_state()

    def record_injection_attempt(self):
        """Record a detected injection attempt."""
        self.injection_attempts += 1
        self.anomaly_score = min(1.0, self.anomaly_score + 0.3)
        self._update_state()

    def update_coherence(self, score: float):
        """Update the logical coherence score."""
        self.logical_coherence_score = max(0.0, min(1.0, score))
        self._update_state()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "trust_level": self.trust_level.name,
            "anomaly_score": self.anomaly_score,
            "manipulation_attempts": self.manipulation_attempts,
            "injection_attempts": self.injection_attempts,
            "logical_coherence_score": self.logical_coherence_score,
            "message_count": len(self.history),
            "created_at": self.created_at.isoformat()
        }
