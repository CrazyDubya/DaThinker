"""Base agent class for thinking agents."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..openrouter import OpenRouterClient, Message
from ..security import sanitize_input, get_security_prefix

if TYPE_CHECKING:
    pass


class AgentRole(str, Enum):
    """Roles that agents can play in helping users think."""
    SOCRATIC = "socratic"  # Asks probing questions
    DEVILS_ADVOCATE = "devils_advocate"  # Challenges assumptions
    CLARIFIER = "clarifier"  # Identifies ambiguities, defines terms
    SYNTHESIZER = "synthesizer"  # Integrates ideas, finds patterns
    PERSPECTIVE = "perspective"  # Offers alternative viewpoints


class AgentIntent(str, Enum):
    """The intent behind an agent's response."""
    CLARIFY = "clarify"      # Seeking to clarify definitions or meaning
    CHALLENGE = "challenge"  # Challenging assumptions or claims
    EXPAND = "expand"        # Expanding perspective or options
    SYNTHESIZE = "synthesize"  # Connecting or organizing ideas
    QUESTION = "question"    # Probing with questions
    VALIDATE = "validate"    # Confirming or supporting a point


@dataclass
class TargetedElement:
    """An element the agent is targeting (assumption, claim, definition)."""
    element_type: str  # "assumption", "claim", "definition", "framing"
    content: str
    action: str  # "question", "challenge", "clarify", "support"


@dataclass
class AgentResponse:
    """Response from a thinking agent.

    Structured for machine-usability while maintaining readable content.
    """
    agent_name: str
    role: AgentRole
    content: str

    # Core structured outputs
    questions: list[str] = field(default_factory=list)  # Questions to prompt further thinking
    challenges: list[str] = field(default_factory=list)  # Challenges to assumptions
    insights: list[str] = field(default_factory=list)  # Key insights identified

    # Enhanced structured outputs (D from spec)
    intent: AgentIntent = AgentIntent.QUESTION  # Primary intent of this response
    targets: list[TargetedElement] = field(default_factory=list)  # What's being acted upon
    proposals: list[str] = field(default_factory=list)  # Concrete proposals or suggestions
    citations: list[str] = field(default_factory=list)  # Source citations (for future retrieval)

    # Metadata
    security_warnings: list[str] = field(default_factory=list)  # Any security warnings

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "role": self.role.value,
            "content": self.content,
            "intent": self.intent.value,
            "questions": self.questions,
            "challenges": self.challenges,
            "insights": self.insights,
            "proposals": self.proposals,
            "targets": [
                {"type": t.element_type, "content": t.content, "action": t.action}
                for t in self.targets
            ],
            "citations": self.citations,
            "security_warnings": self.security_warnings,
        }


class BaseAgent(ABC):
    """Base class for all thinking agents.

    Each agent has a specific role in helping users think deeper:
    - They don't give answers, they prompt reflection
    - They challenge, question, and expand thinking
    - They help users arrive at their own conclusions
    """

    def __init__(
        self,
        client: OpenRouterClient,
        model: str = "balanced",
        temperature: float = 0.7,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.conversation_history: list[Message] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent's display name."""
        pass

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Agent's role in the thinking process."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining agent behavior."""
        pass

    @property
    def secure_system_prompt(self) -> str:
        """System prompt with security prefix."""
        return get_security_prefix() + self.system_prompt

    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []

    async def think(
        self,
        user_input: str,
        context: str | None = None,
        other_agents_input: list["AgentResponse"] | None = None,
    ) -> AgentResponse:
        """Process user input and generate a response that promotes deeper thinking.

        Args:
            user_input: The user's current statement or question
            context: Optional context from the conversation
            other_agents_input: Optional responses from other agents

        Returns:
            AgentResponse with questions, challenges, and insights
        """
        # Sanitize user input for security
        sanitization = sanitize_input(user_input)
        safe_input = sanitization.sanitized_input

        messages = [Message(role="system", content=self.secure_system_prompt)]

        # Add context if provided (as system context, not user message)
        if context:
            messages.append(Message(
                role="system",
                content=f"Session Context:\n{context}"
            ))

        # Add other agents' perspectives as assistant messages to prevent confusion
        if other_agents_input:
            for r in other_agents_input:
                # Each agent's prior response goes as assistant, avoiding user role confusion
                messages.append(Message(
                    role="assistant",
                    content=f"[Previous analysis by {r.agent_name}]:\n{r.content}"
                ))
                messages.append(Message(
                    role="user",
                    content="Please provide your perspective on the user's original question, building on but not repeating the above analysis."
                ))

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current user input
        messages.append(Message(role="user", content=safe_input))

        # Get response from LLM
        response_content = await self.client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
        )

        # Clean response - remove any leaked internal markers
        cleaned_response = self._clean_response(response_content)

        # Store in history (store original input, not sanitized)
        self.conversation_history.append(Message(role="user", content=user_input))
        self.conversation_history.append(Message(role="assistant", content=cleaned_response))

        # Parse and return structured response
        response = self._parse_response(cleaned_response)
        response.security_warnings = sanitization.warnings
        return response

    def _clean_response(self, content: str) -> str:
        """Remove any leaked internal markers from response."""
        # Remove leaked context markers
        patterns_to_remove = [
            r"\[OTHER PERSPECTIVES\].*?(?=\n\n|\Z)",
            r"\[Previous analysis by \w+\]:.*?(?=\n\n|\Z)",
            r"\[USER MESSAGE.*?\]",
            r"\[END USER MESSAGE\]",
            r"\[CONTEXT\]",
            r"\[S Y S T E M\]",
        ]

        cleaned = content
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def _parse_response(self, content: str) -> AgentResponse:
        """Parse LLM response into structured AgentResponse.

        Enhanced parsing that extracts intent, targets, and proposals.
        """
        questions = []
        challenges = []
        insights = []
        proposals = []
        targets = []

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Detect section headers (various formats)
            if any(q in line_lower for q in ["question", "asking", "consider", "explore"]):
                if line.startswith("#") or line.startswith("**") or line.endswith(":"):
                    current_section = "questions"
            elif any(c in line_lower for c in ["challenge", "counter", "weak", "assumption"]):
                if line.startswith("#") or line.startswith("**") or line.endswith(":"):
                    current_section = "challenges"
            elif any(i in line_lower for i in ["insight", "pattern", "observation", "key point"]):
                if line.startswith("#") or line.startswith("**") or line.endswith(":"):
                    current_section = "insights"
            elif any(p in line_lower for p in ["suggest", "propose", "recommend", "could", "might"]):
                if line.startswith("#") or line.startswith("**") or line.endswith(":"):
                    current_section = "proposals"

            # Extract bullet points
            if line_stripped.startswith(("- ", "* ", "• ")):
                item = line_stripped[2:].strip()
                if item:
                    if current_section == "questions":
                        questions.append(item)
                    elif current_section == "challenges":
                        challenges.append(item)
                    elif current_section == "insights":
                        insights.append(item)
                    elif current_section == "proposals":
                        proposals.append(item)

            # Also extract numbered items
            numbered_match = re.match(r'^\d+[\.\)]\s*(.+)', line_stripped)
            if numbered_match:
                item = numbered_match.group(1).strip()
                if item and current_section:
                    if current_section == "questions":
                        questions.append(item)
                    elif current_section == "challenges":
                        challenges.append(item)
                    elif current_section == "insights":
                        insights.append(item)
                    elif current_section == "proposals":
                        proposals.append(item)

            # Extract questions by pattern (sentences ending in ?)
            if "?" in line_stripped and current_section == "questions":
                # Extract the question
                q_match = re.search(r'([^.!?]*\?)', line_stripped)
                if q_match and q_match.group(1) not in questions:
                    questions.append(q_match.group(1).strip())

        # Determine primary intent based on role and content
        intent = self._determine_intent(content, questions, challenges, insights)

        # Extract targeted elements (assumptions, claims, definitions being acted upon)
        targets = self._extract_targets(content)

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            content=content,
            questions=questions,
            challenges=challenges,
            insights=insights,
            intent=intent,
            targets=targets,
            proposals=proposals,
            citations=[],  # Will be populated when retrieval is added
            security_warnings=[],
        )

    def _determine_intent(
        self,
        content: str,
        questions: list[str],
        challenges: list[str],
        insights: list[str],
    ) -> "AgentIntent":
        """Determine the primary intent of the response."""
        from .base import AgentIntent

        content_lower = content.lower()

        # Role-based default intents
        role_intents = {
            AgentRole.SOCRATIC: AgentIntent.QUESTION,
            AgentRole.DEVILS_ADVOCATE: AgentIntent.CHALLENGE,
            AgentRole.CLARIFIER: AgentIntent.CLARIFY,
            AgentRole.SYNTHESIZER: AgentIntent.SYNTHESIZE,
            AgentRole.PERSPECTIVE: AgentIntent.EXPAND,
        }

        # Start with role default
        intent = role_intents.get(self.role, AgentIntent.QUESTION)

        # Adjust based on content analysis
        if len(questions) > len(challenges) and len(questions) > len(insights):
            intent = AgentIntent.QUESTION
        elif len(challenges) > len(questions) and len(challenges) > len(insights):
            intent = AgentIntent.CHALLENGE
        elif "what if" in content_lower or "consider" in content_lower or "alternatively" in content_lower:
            intent = AgentIntent.EXPAND
        elif "therefore" in content_lower or "in summary" in content_lower or "connecting" in content_lower:
            intent = AgentIntent.SYNTHESIZE
        elif "define" in content_lower or "what do you mean" in content_lower or "unclear" in content_lower:
            intent = AgentIntent.CLARIFY

        return intent

    def _extract_targets(self, content: str) -> list["TargetedElement"]:
        """Extract targeted elements (assumptions, claims, definitions) from content."""
        from .base import TargetedElement

        targets = []
        content_lower = content.lower()

        # Patterns for identifying targeted elements
        patterns = [
            # Assumptions
            (r"(?:you(?:'re| are) assuming|assumption that|assumes that)\s+(.+?)(?:\.|,|$)", "assumption", "question"),
            (r"(?:underlying assumption|implicit assumption):\s*(.+?)(?:\.|$)", "assumption", "question"),

            # Claims
            (r"(?:your claim that|the claim that|you(?:'re| are) claiming)\s+(.+?)(?:\.|,|$)", "claim", "challenge"),
            (r"(?:this suggests|this implies)\s+(.+?)(?:\.|$)", "claim", "question"),

            # Definitions
            (r"(?:what do you mean by|define)\s+['\"]?(.+?)['\"]?(?:\?|$)", "definition", "clarify"),
            (r"(?:the term|the word)\s+['\"]?(.+?)['\"]?\s+(?:is|seems)", "definition", "clarify"),

            # Framing
            (r"(?:you(?:'re| are) framing|framed as|the framing)\s+(.+?)(?:\.|,|$)", "framing", "challenge"),
        ]

        for pattern, element_type, action in patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches[:2]:  # Limit to 2 per type
                if len(match) > 5:  # Meaningful content
                    targets.append(TargetedElement(
                        element_type=element_type,
                        content=match.strip()[:100],  # Truncate long matches
                        action=action,
                    ))

        return targets
