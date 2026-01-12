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


@dataclass
class AgentResponse:
    """Response from a thinking agent."""
    agent_name: str
    role: AgentRole
    content: str
    questions: list[str] = field(default_factory=list)  # Questions to prompt further thinking
    challenges: list[str] = field(default_factory=list)  # Challenges to assumptions
    insights: list[str] = field(default_factory=list)  # Key insights identified
    security_warnings: list[str] = field(default_factory=list)  # Any security warnings


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

        Improved parsing that handles various formats.
        """
        questions = []
        challenges = []
        insights = []

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

            # Extract questions by pattern (sentences ending in ?)
            if "?" in line_stripped and current_section == "questions":
                # Extract the question
                q_match = re.search(r'([^.!?]*\?)', line_stripped)
                if q_match and q_match.group(1) not in questions:
                    questions.append(q_match.group(1).strip())

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            content=content,
            questions=questions,
            challenges=challenges,
            insights=insights,
            security_warnings=[],
        )
