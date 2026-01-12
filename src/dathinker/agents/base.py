"""Base agent class for thinking agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..openrouter import OpenRouterClient, Message

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

    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []

    async def think(
        self,
        user_input: str,
        context: str | None = None,
        other_agents_input: list[AgentResponse] | None = None,
    ) -> AgentResponse:
        """Process user input and generate a response that promotes deeper thinking.

        Args:
            user_input: The user's current statement or question
            context: Optional context from the conversation
            other_agents_input: Optional responses from other agents

        Returns:
            AgentResponse with questions, challenges, and insights
        """
        messages = [Message(role="system", content=self.system_prompt)]

        # Add context if provided
        if context:
            messages.append(Message(
                role="user",
                content=f"[CONTEXT]\n{context}"
            ))

        # Add other agents' perspectives if available
        if other_agents_input:
            agent_perspectives = "\n\n".join([
                f"[{r.agent_name} ({r.role.value})]:\n{r.content}"
                for r in other_agents_input
            ])
            messages.append(Message(
                role="user",
                content=f"[OTHER PERSPECTIVES]\n{agent_perspectives}"
            ))

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current user input
        messages.append(Message(role="user", content=user_input))

        # Get response from LLM
        response_content = await self.client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
        )

        # Store in history
        self.conversation_history.append(Message(role="user", content=user_input))
        self.conversation_history.append(Message(role="assistant", content=response_content))

        # Parse and return structured response
        return self._parse_response(response_content)

    def _parse_response(self, content: str) -> AgentResponse:
        """Parse LLM response into structured AgentResponse.

        Subclasses can override for custom parsing.
        """
        # Simple parsing - look for sections marked with headers
        questions = []
        challenges = []
        insights = []

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()
            if "question" in line_lower and line.startswith("#"):
                current_section = "questions"
            elif "challenge" in line_lower and line.startswith("#"):
                current_section = "challenges"
            elif "insight" in line_lower and line.startswith("#"):
                current_section = "insights"
            elif line.strip().startswith("- ") or line.strip().startswith("* "):
                item = line.strip()[2:].strip()
                if current_section == "questions":
                    questions.append(item)
                elif current_section == "challenges":
                    challenges.append(item)
                elif current_section == "insights":
                    insights.append(item)

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            content=content,
            questions=questions,
            challenges=challenges,
            insights=insights,
        )
