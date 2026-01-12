"""Orchestrator for multi-agent thinking sessions."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Optional

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
from .router import (
    BaseRouter,
    RouterType,
    RoutingTrace,
    create_router,
)


class ThinkingMode(str, Enum):
    """Different modes for how agents collaborate."""
    SINGLE = "single"  # One agent at a time, user chooses
    ROUND_ROBIN = "round_robin"  # Each agent responds in turn
    PARALLEL = "parallel"  # All agents respond simultaneously
    ADAPTIVE = "adaptive"  # Orchestrator chooses based on context


@dataclass
class Assumption:
    """A working assumption pinned during the session."""
    id: int
    content: str
    status: str = "open"  # open, confirmed, contested, resolved
    turn_added: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class Constraint:
    """A constraint that agents must respect."""
    id: int
    content: str
    category: str = "general"  # time, money, ethics, scope, technical


@dataclass
class ThinkingSession:
    """A thinking session with history and context."""
    id: str
    topic: str
    mode: ThinkingMode
    history: list[dict] = field(default_factory=list)  # Full conversation history
    insights: list[str] = field(default_factory=list)  # Accumulated insights
    questions: list[str] = field(default_factory=list)  # Open questions
    # Session control surfaces (v0.2)
    assumptions: list[Assumption] = field(default_factory=list)  # Pinned assumptions
    goal: str = ""  # What we're optimizing for
    constraints: list[Constraint] = field(default_factory=list)  # Constraints to respect
    routing_traces: list[RoutingTrace] = field(default_factory=list)  # Routing history


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
        router_type: RouterType = RouterType.HEURISTIC,
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

        # Initialize router (v0.2 - pluggable routing)
        self.router_type = router_type
        self.router: BaseRouter = create_router(router_type, self.client)

        self.active_session: ThinkingSession | None = None
        self._session_counter = 0
        self._assumption_counter = 0
        self._constraint_counter = 0

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
        on_routing_trace: Callable[[RoutingTrace], Awaitable[None]] | None = None,
    ) -> tuple[list[AgentResponse], RoutingTrace]:
        """Adaptively choose and sequence agents based on the input.

        This is the smart mode that analyzes the input and decides which
        agents should respond and in what order.

        Returns:
            Tuple of (responses, routing_trace) for full explainability.
        """
        # Get turn count for router
        turn_count = len([h for h in (self.active_session.history if self.active_session else []) if h.get("type") == "user"])

        # Build context including session goal and constraints
        full_context = context or ""
        if self.active_session:
            if self.active_session.goal:
                full_context = f"GOAL: {self.active_session.goal}\n{full_context}"
            if self.active_session.constraints:
                constraints_text = "\n".join(f"- {c.content}" for c in self.active_session.constraints)
                full_context = f"CONSTRAINTS:\n{constraints_text}\n{full_context}"
            if self.active_session.assumptions:
                open_assumptions = [a for a in self.active_session.assumptions if a.status == "open"]
                if open_assumptions:
                    assumptions_text = "\n".join(f"- {a.content}" for a in open_assumptions)
                    full_context = f"WORKING ASSUMPTIONS:\n{assumptions_text}\n{full_context}"

        # Use pluggable router for agent selection
        routing_trace = await self.router.route(
            user_input=user_input,
            context=full_context if full_context else None,
            session_history=self.active_session.history if self.active_session else None,
            turn_count=turn_count,
        )

        # Notify about routing decision
        if on_routing_trace:
            await on_routing_trace(routing_trace)

        responses: list[AgentResponse] = []

        # Get responses in the suggested order
        for agent_name in routing_trace.selection_order:
            agent = self.agents.get(agent_name)
            if not agent:
                continue

            response = await agent.think(
                user_input,
                full_context if full_context else None,
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
            self.active_session.routing_traces.append(routing_trace)
            for response in responses:
                self.active_session.history.append({
                    "type": "agent",
                    "agent": response.agent_name,
                    "response": response,
                })
                self.active_session.insights.extend(response.insights)
                self.active_session.questions.extend(response.questions)

        return responses, routing_trace

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

    async def synthesize_session(
        self,
        style: str = "default",
    ) -> dict:
        """Generate a multi-level synthesis of the current thinking session.

        Args:
            style: Output style - 'default', 'memo', 'outline', 'debate', 'todo'

        Returns:
            Dict with tldr, map, and next_moves sections.
        """
        if not self.active_session or not self.active_session.history:
            return {
                "tldr": ["No active session or empty history."],
                "map": {},
                "next_moves": [],
                "style": style,
            }

        # Build context from session
        history_text = []
        for entry in self.active_session.history:
            if entry["type"] == "user":
                history_text.append(f"USER: {entry['content']}")
            else:
                history_text.append(f"{entry['agent'].upper()}: {entry['response'].content}")

        # Include assumptions and constraints context
        context_parts = []
        if self.active_session.goal:
            context_parts.append(f"Session Goal: {self.active_session.goal}")
        if self.active_session.assumptions:
            assumptions_text = "\n".join(
                f"- [{a.status.upper()}] {a.content}"
                for a in self.active_session.assumptions
            )
            context_parts.append(f"Working Assumptions:\n{assumptions_text}")
        if self.active_session.constraints:
            constraints_text = "\n".join(
                f"- [{c.category}] {c.content}"
                for c in self.active_session.constraints
            )
            context_parts.append(f"Constraints:\n{constraints_text}")

        context_section = "\n\n".join(context_parts) if context_parts else ""

        style_instructions = {
            "default": "Use clear sections with bullet points.",
            "memo": "Format as a professional decision memo with Executive Summary, Analysis, Recommendation sections.",
            "outline": "Use hierarchical outline format with Roman numerals and sub-points.",
            "debate": "Present as a structured debate with Pro/Con/Resolution sections.",
            "todo": "Format as actionable checklist items with priorities and owners.",
        }

        synthesis_prompt = f"""Analyze this thinking session and provide a structured synthesis.

Topic: {self.active_session.topic}
{context_section}

Session history:
{chr(10).join(history_text)}

Provide a JSON response with this exact structure:
{{
    "tldr": [
        "bullet 1 (most important insight)",
        "bullet 2",
        "bullet 3",
        "bullet 4 (if needed)",
        "bullet 5 (if needed)"
    ],
    "map": {{
        "key_claims": ["claim 1", "claim 2"],
        "evidence": ["evidence 1", "evidence 2"],
        "assumptions": ["assumption 1", "assumption 2"],
        "open_questions": ["question 1", "question 2"],
        "conflicts": ["conflict between agents or ideas"]
    }},
    "next_moves": [
        {{"action": "description", "type": "experiment|decision|question", "priority": "high|medium|low"}},
        {{"action": "description", "type": "experiment|decision|question", "priority": "high|medium|low"}}
    ]
}}

Style: {style_instructions.get(style, style_instructions['default'])}

Return ONLY valid JSON, no markdown code blocks or other text."""

        messages = [Message(role="user", content=synthesis_prompt)]

        try:
            response = await self.client.chat(
                messages=messages,
                model=self.model,
                temperature=0.4,
                max_tokens=2000,
            )

            # Parse JSON response
            import json
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)
            result["style"] = style
            return result

        except (json.JSONDecodeError, Exception) as e:
            # Fallback to simple text synthesis
            return {
                "tldr": [f"Synthesis generation encountered an issue: {str(e)[:50]}"],
                "map": {
                    "key_claims": [],
                    "evidence": [],
                    "assumptions": [a.content for a in self.active_session.assumptions],
                    "open_questions": list(set(self.active_session.questions))[:5],
                    "conflicts": [],
                },
                "next_moves": [],
                "style": style,
                "raw_response": response if 'response' in dir() else None,
            }

    # ===== Session Control Surfaces (v0.2) =====

    def set_router(self, router_type: RouterType) -> None:
        """Switch to a different router implementation."""
        self.router_type = router_type
        self.router = create_router(router_type, self.client)

    def get_router_info(self) -> dict:
        """Get information about the current router."""
        return {
            "type": self.router_type.value,
            "description": {
                RouterType.HEURISTIC: "Fast pattern-based routing (deterministic)",
                RouterType.LLM: "LLM-based intelligent routing",
                RouterType.HYBRID: "Heuristic + LLM tie-break",
            }.get(self.router_type, "Unknown"),
        }

    def pin_assumption(self, content: str) -> Assumption:
        """Pin a statement as a working assumption."""
        if not self.active_session:
            raise ValueError("No active session")

        self._assumption_counter += 1
        turn = len([h for h in self.active_session.history if h.get("type") == "user"])

        assumption = Assumption(
            id=self._assumption_counter,
            content=content,
            status="open",
            turn_added=turn,
        )
        self.active_session.assumptions.append(assumption)
        return assumption

    def update_assumption(
        self,
        assumption_id: int,
        status: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Assumption | None:
        """Update an assumption's status or add a note."""
        if not self.active_session:
            return None

        for assumption in self.active_session.assumptions:
            if assumption.id == assumption_id:
                if status:
                    assumption.status = status
                if note:
                    assumption.notes.append(note)
                return assumption
        return None

    def get_assumptions(self, status_filter: Optional[str] = None) -> list[Assumption]:
        """Get all assumptions, optionally filtered by status."""
        if not self.active_session:
            return []

        if status_filter:
            return [a for a in self.active_session.assumptions if a.status == status_filter]
        return self.active_session.assumptions

    def set_goal(self, goal: str) -> None:
        """Set the session's optimization goal."""
        if not self.active_session:
            raise ValueError("No active session")
        self.active_session.goal = goal

    def get_goal(self) -> str:
        """Get the current session goal."""
        if not self.active_session:
            return ""
        return self.active_session.goal

    def add_constraint(self, content: str, category: str = "general") -> Constraint:
        """Add a constraint that agents must respect."""
        if not self.active_session:
            raise ValueError("No active session")

        valid_categories = ["time", "money", "ethics", "scope", "technical", "general"]
        if category not in valid_categories:
            category = "general"

        self._constraint_counter += 1
        constraint = Constraint(
            id=self._constraint_counter,
            content=content,
            category=category,
        )
        self.active_session.constraints.append(constraint)
        return constraint

    def remove_constraint(self, constraint_id: int) -> bool:
        """Remove a constraint by ID."""
        if not self.active_session:
            return False

        original_len = len(self.active_session.constraints)
        self.active_session.constraints = [
            c for c in self.active_session.constraints if c.id != constraint_id
        ]
        return len(self.active_session.constraints) < original_len

    def get_constraints(self, category_filter: Optional[str] = None) -> list[Constraint]:
        """Get all constraints, optionally filtered by category."""
        if not self.active_session:
            return []

        if category_filter:
            return [c for c in self.active_session.constraints if c.category == category_filter]
        return self.active_session.constraints

    def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        if not self.active_session:
            return {"error": "No active session"}

        return {
            "id": self.active_session.id,
            "topic": self.active_session.topic,
            "mode": self.active_session.mode.value,
            "router": self.router_type.value,
            "turns": len([h for h in self.active_session.history if h["type"] == "user"]),
            "unique_insights": len(set(self.active_session.insights)),
            "open_questions": len(set(self.active_session.questions)),
            "goal": self.active_session.goal or "(not set)",
            "assumptions": {
                "total": len(self.active_session.assumptions),
                "open": len([a for a in self.active_session.assumptions if a.status == "open"]),
                "confirmed": len([a for a in self.active_session.assumptions if a.status == "confirmed"]),
                "contested": len([a for a in self.active_session.assumptions if a.status == "contested"]),
            },
            "constraints": len(self.active_session.constraints),
            "routing_decisions": len(self.active_session.routing_traces),
        }
