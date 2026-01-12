"""Orchestrator for multi-agent thinking sessions."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable

from .openrouter import OpenRouterClient, Message
from .agents import (
    BaseAgent,
    AgentResponse,
    SocraticAgent,
    DevilsAdvocateAgent,
    ClarifierAgent,
    SynthesizerAgent,
    PerspectiveExpanderAgent,
)


class ThinkingMode(str, Enum):
    """Different modes for how agents collaborate."""
    SINGLE = "single"  # One agent at a time, user chooses
    ROUND_ROBIN = "round_robin"  # Each agent responds in turn
    PARALLEL = "parallel"  # All agents respond simultaneously
    ADAPTIVE = "adaptive"  # Orchestrator chooses based on context


@dataclass
class ThinkingSession:
    """A thinking session with history and context."""
    id: str
    topic: str
    mode: ThinkingMode
    history: list[dict] = field(default_factory=list)  # Full conversation history
    insights: list[str] = field(default_factory=list)  # Accumulated insights
    questions: list[str] = field(default_factory=list)  # Open questions


class ThinkingOrchestrator:
    """Orchestrates multiple thinking agents to help users think deeper.

    The orchestrator:
    - Manages a panel of thinking agents
    - Coordinates their interactions based on the thinking mode
    - Maintains session context and history
    - Synthesizes agent outputs into coherent dialogue
    """

    def __init__(
        self,
        client: OpenRouterClient | None = None,
        model: str = "balanced",
    ):
        self.client = client or OpenRouterClient()
        self.model = model

        # Initialize all agents
        self.agents: dict[str, BaseAgent] = {
            "socratic": SocraticAgent(self.client, model),
            "advocate": DevilsAdvocateAgent(self.client, model),
            "clarifier": ClarifierAgent(self.client, model),
            "synthesizer": SynthesizerAgent(self.client, model),
            "expander": PerspectiveExpanderAgent(self.client, model),
        }

        self.active_session: ThinkingSession | None = None
        self._session_counter = 0

    def start_session(
        self,
        topic: str,
        mode: ThinkingMode = ThinkingMode.ADAPTIVE,
    ) -> ThinkingSession:
        """Start a new thinking session on a topic."""
        self._session_counter += 1
        session = ThinkingSession(
            id=f"session_{self._session_counter}",
            topic=topic,
            mode=mode,
        )
        self.active_session = session

        # Reset all agents for new session
        for agent in self.agents.values():
            agent.reset()

        return session

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get a specific agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """List available agent names."""
        return list(self.agents.keys())

    async def think_with_agent(
        self,
        user_input: str,
        agent_name: str,
        context: str | None = None,
    ) -> AgentResponse:
        """Get response from a specific agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")

        response = await agent.think(user_input, context)

        # Record in session history
        if self.active_session:
            self.active_session.history.append({
                "type": "user",
                "content": user_input,
            })
            self.active_session.history.append({
                "type": "agent",
                "agent": agent_name,
                "response": response,
            })
            # Accumulate insights and questions
            self.active_session.insights.extend(response.insights)
            self.active_session.questions.extend(response.questions)

        return response

    async def think_parallel(
        self,
        user_input: str,
        agent_names: list[str] | None = None,
        context: str | None = None,
    ) -> list[AgentResponse]:
        """Get responses from multiple agents in parallel."""
        agent_names = agent_names or list(self.agents.keys())

        # Run all agents in parallel
        tasks = [
            self.agents[name].think(user_input, context)
            for name in agent_names
            if name in self.agents
        ]

        responses = await asyncio.gather(*tasks)

        # Record in session history
        if self.active_session:
            self.active_session.history.append({
                "type": "user",
                "content": user_input,
            })
            for response in responses:
                self.active_session.history.append({
                    "type": "agent",
                    "agent": response.agent_name,
                    "response": response,
                })
                self.active_session.insights.extend(response.insights)
                self.active_session.questions.extend(response.questions)

        return list(responses)

    async def think_adaptive(
        self,
        user_input: str,
        context: str | None = None,
        on_agent_response: Callable[[AgentResponse], Awaitable[None]] | None = None,
    ) -> list[AgentResponse]:
        """Adaptively choose and sequence agents based on the input.

        This is the smart mode that analyzes the input and decides which
        agents should respond and in what order.
        """
        # First, determine which agents should respond
        agent_selection = await self._select_agents(user_input, context)

        responses: list[AgentResponse] = []

        # Get responses in the suggested order
        for agent_name in agent_selection:
            agent = self.agents.get(agent_name)
            if not agent:
                continue

            response = await agent.think(
                user_input,
                context,
                other_agents_input=responses if responses else None,
            )
            responses.append(response)

            if on_agent_response:
                await on_agent_response(response)

        # Record in session history
        if self.active_session:
            self.active_session.history.append({
                "type": "user",
                "content": user_input,
            })
            for response in responses:
                self.active_session.history.append({
                    "type": "agent",
                    "agent": response.agent_name,
                    "response": response,
                })
                self.active_session.insights.extend(response.insights)
                self.active_session.questions.extend(response.questions)

        return responses

    async def _select_agents(
        self,
        user_input: str,
        context: str | None = None,
    ) -> list[str]:
        """Use LLM to determine which agents should respond and in what order.

        Returns a list of agent names in suggested order.
        """
        selection_prompt = f"""You are an orchestrator for a thinking assistance system. Given user input, decide which thinking agents should respond and in what order.

Available agents:
- socratic: Asks probing questions to deepen understanding. Use when the user is making claims or needs to examine their beliefs.
- advocate: Challenges assumptions and presents counterarguments. Use when the user seems certain or has clear positions.
- clarifier: Identifies ambiguity and asks for precise definitions. Use when terms are vague or the statement is unclear.
- synthesizer: Finds patterns and organizes ideas. Use when there's complexity to organize or multiple ideas to connect.
- expander: Offers alternative perspectives. Use when thinking seems narrow or could benefit from other viewpoints.

User input: {user_input}
{f"Context: {context}" if context else ""}

Respond with ONLY a comma-separated list of 2-3 agent names in the order they should respond. Always include at least 2 agents to provide multiple perspectives. Choose based on what would most help the user think deeper. Example: "socratic,expander" or "clarifier,socratic,advocate"

Agents to use:"""

        messages = [Message(role="user", content=selection_prompt)]

        response = await self.client.chat(
            messages=messages,
            model="fast",  # Use fast model for meta-decisions
            temperature=0.3,
            max_tokens=50,
        )

        # Parse response
        agent_names = [
            name.strip().lower()
            for name in response.strip().split(",")
        ]

        # Validate and filter to known agents
        valid_agents = [
            name for name in agent_names
            if name in self.agents
        ]

        # Default to socratic if no valid agents
        return valid_agents if valid_agents else ["socratic"]

    async def synthesize_session(self) -> str:
        """Generate a synthesis of the current thinking session."""
        if not self.active_session or not self.active_session.history:
            return "No active session or empty history."

        # Build context from session
        history_text = []
        for entry in self.active_session.history:
            if entry["type"] == "user":
                history_text.append(f"USER: {entry['content']}")
            else:
                history_text.append(f"{entry['agent'].upper()}: {entry['response'].content}")

        synthesis_prompt = f"""Review this thinking session and provide a synthesis that helps the user see their progress and remaining questions.

Topic: {self.active_session.topic}

Session history:
{chr(10).join(history_text)}

Provide:
1. Key insights that emerged
2. How the user's thinking evolved
3. Important questions still open
4. Suggested next steps for continued thinking

Be concise but substantive. Focus on helping them see their own progress."""

        messages = [Message(role="user", content=synthesis_prompt)]

        return await self.client.chat(
            messages=messages,
            model=self.model,
            temperature=0.5,
        )

    def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        if not self.active_session:
            return {"error": "No active session"}

        return {
            "id": self.active_session.id,
            "topic": self.active_session.topic,
            "mode": self.active_session.mode.value,
            "turns": len([h for h in self.active_session.history if h["type"] == "user"]),
            "unique_insights": len(set(self.active_session.insights)),
            "open_questions": len(set(self.active_session.questions)),
        }
