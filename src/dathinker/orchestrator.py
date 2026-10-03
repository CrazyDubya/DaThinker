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


class AssumptionStatus(str, Enum):
    """Status of a pinned assumption."""
    OPEN = "open"
    CONFIRMED = "confirmed"
    CONTESTED = "contested"
    REVISED = "revised"


class SynthesisStyle(str, Enum):
    """Output style for synthesis."""
    MEMO = "memo"
    OUTLINE = "outline"
    DEBATE = "debate"
    TODO = "todo"


@dataclass
class PinnedStatement:
    """A user-pinned statement or assumption (string id form)."""
    id: str
    content: str
    status: AssumptionStatus = AssumptionStatus.OPEN
    turn_created: int = 0
    turn_last_referenced: int = 0
    agent_references: list[str] = field(default_factory=list)


@dataclass
class Goal:
    """A session goal the user is optimizing for."""
    id: str
    content: str
    priority: int = 1  # 1 = highest
    active: bool = True


@dataclass
class MultiLevelSynthesis:
    """Multi-level synthesis output."""
    tldr: list[str]
    key_claims: list[str]
    evidence: list[str]
    assumptions: list[str]
    open_questions: list[str]
    conflicts: list[str]
    next_moves: list[str]
    raw_text: str


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
    hard: bool = True  # Hard = must respect, Soft = prefer to respect


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
    # Richer session surfaces (multi-goal, string-id pins) alongside the v0.2 fields above.
    pins: list[PinnedStatement] = field(default_factory=list)
    goals: list[Goal] = field(default_factory=list)
    _pin_counter: int = 0
    _goal_counter: int = 0


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
        router_version: RouterType | None = None,
    ):
        if router_version is not None:
            router_type = router_version
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
        self.router_version = router_type
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
            raw = response if isinstance(locals().get("response"), str) else ""
            if "##" in raw:
                parsed = self._parse_synthesis(raw)
                if isinstance(style, SynthesisStyle) or parsed.tldr or parsed.key_claims or parsed.next_moves:
                    return parsed
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
        self.router_version = router_type
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

    def add_constraint(self, content: str, category: str = "general", hard: bool = True) -> Constraint:
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
            hard=hard,
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
            "pins": len(self.active_session.pins),
            "goals": len([g for g in self.active_session.goals if g.active]),
            "routing_decisions": len(self.active_session.routing_traces),
            "routing_traces": len(self.active_session.routing_traces),
        }

    def pin(self, content: str) -> PinnedStatement:
        """Pin a statement as a working assumption (string-id form)."""
        if not self.active_session:
            raise ValueError("No active session")

        self.active_session._pin_counter += 1
        turn = len([h for h in self.active_session.history if h.get("type") == "user"])
        pin = PinnedStatement(
            id=f"pin_{self.active_session._pin_counter}",
            content=content,
            turn_created=turn,
            turn_last_referenced=turn,
        )
        self.active_session.pins.append(pin)
        return pin

    def get_pins(self, status: AssumptionStatus | None = None) -> list[PinnedStatement]:
        """Get all pins, optionally filtered by status."""
        if not self.active_session:
            return []
        pins = self.active_session.pins
        if status:
            pins = [p for p in pins if p.status == status]
        return pins

    def update_pin_status(self, pin_id: str, status: AssumptionStatus) -> bool:
        """Update the status of a pinned statement."""
        if not self.active_session:
            return False
        for pin in self.active_session.pins:
            if pin.id == pin_id:
                pin.status = status
                return True
        return False

    def add_goal(self, content: str, priority: int = 1) -> Goal:
        """Add a session goal."""
        if not self.active_session:
            raise ValueError("No active session")
        self.active_session._goal_counter += 1
        goal = Goal(
            id=f"goal_{self.active_session._goal_counter}",
            content=content,
            priority=priority,
        )
        self.active_session.goals.append(goal)
        self.active_session.goals.sort(key=lambda g: g.priority)
        return goal

    def get_goals(self, active_only: bool = True) -> list[Goal]:
        """Get session goals."""
        if not self.active_session:
            return []
        goals = self.active_session.goals
        if active_only:
            goals = [g for g in goals if g.active]
        return goals

    def deactivate_goal(self, goal_id: str) -> bool:
        """Mark a goal as inactive."""
        if not self.active_session:
            return False
        for goal in self.active_session.goals:
            if goal.id == goal_id:
                goal.active = False
                return True
        return False

    def _build_context_for_agents(self) -> str:
        """Build context string including goals, constraints, and pins."""
        if not self.active_session:
            return ""
        parts = []
        goals = self.get_goals(active_only=True)
        if goals:
            parts.append("SESSION GOALS:\n" + "\n".join(f"- {g.content}" for g in goals))
        elif self.active_session.goal:
            parts.append(f"SESSION GOALS:\n- {self.active_session.goal}")
        constraints = self.get_constraints()
        if constraints:
            parts.append(
                "CONSTRAINTS:\n"
                + "\n".join(
                    f"- {'[HARD]' if c.hard else '[SOFT]'} {c.content}" for c in constraints
                )
            )
        pins = self.get_pins()
        if pins:
            parts.append(
                "WORKING ASSUMPTIONS:\n"
                + "\n".join(f"- [{p.status.value}] {p.content}" for p in pins)
            )
        elif self.active_session.assumptions:
            parts.append(
                "WORKING ASSUMPTIONS:\n"
                + "\n".join(
                    f"- [{a.status}] {a.content}" for a in self.active_session.assumptions
                )
            )
        return "\n\n".join(parts)

    def _parse_synthesis(self, raw_text: str) -> MultiLevelSynthesis:
        """Parse a markdown synthesis into structured sections."""
        sections = {
            "tldr": [],
            "key_claims": [],
            "evidence": [],
            "assumptions": [],
            "open_questions": [],
            "conflicts": [],
            "next_moves": [],
        }
        current_section = None
        section_map = {
            "tl;dr": "tldr",
            "tldr": "tldr",
            "key claims": "key_claims",
            "claims": "key_claims",
            "evidence": "evidence",
            "assumptions": "assumptions",
            "open questions": "open_questions",
            "questions": "open_questions",
            "conflicts": "conflicts",
            "tensions": "conflicts",
            "next moves": "next_moves",
            "next steps": "next_moves",
            "action": "next_moves",
        }
        for line in raw_text.split("\n"):
            line = line.strip()
            if line.startswith("##") or line.startswith("**"):
                header = line.replace("#", "").replace("*", "").strip().lower()
                for key, section in section_map.items():
                    if key in header:
                        current_section = section
                        break
            elif line.startswith(("-", "*", "•")) and current_section:
                content = line.lstrip("-*• ").strip()
                if content:
                    sections[current_section].append(content)
            elif line and line[0].isdigit() and current_section:
                import re
                match = re.match(r"^\d+[\.\)]\s*(.+)", line)
                if match:
                    sections[current_section].append(match.group(1).strip())
        return MultiLevelSynthesis(
            tldr=sections["tldr"],
            key_claims=sections["key_claims"],
            evidence=sections["evidence"],
            assumptions=sections["assumptions"],
            open_questions=sections["open_questions"],
            conflicts=sections["conflicts"],
            next_moves=sections["next_moves"],
            raw_text=raw_text,
        )

    async def get_smallest_uncertainty_reducer(self) -> str:
        """Smallest next step that would reduce uncertainty the most."""
        if not self.active_session or not self.active_session.history:
            return "Start by sharing what you're thinking about."

        history_text = []
        for entry in self.active_session.history[-10:]:
            if entry["type"] == "user":
                history_text.append(f"USER: {entry['content']}")
            else:
                history_text.append(f"{entry['agent'].upper()}: {entry['response'].content}")

        questions = list(set(self.active_session.questions))[-5:]
        assumptions = [p.content for p in self.active_session.pins if p.status == AssumptionStatus.OPEN]
        prompt = f"""Based on this thinking session, identify the SINGLE smallest next step that would most reduce uncertainty.

Topic: {self.active_session.topic}

Recent conversation:
{chr(10).join(history_text)}

Open questions: {questions if questions else "None identified yet"}
Untested assumptions: {assumptions if assumptions else "None pinned yet"}

Rules:
- Must be SMALL (can be done in 5-30 minutes)
- Must REDUCE UNCERTAINTY (not just gather more opinions)
- Could be: a quick experiment, a specific question to ask someone, looking up one fact, testing one assumption
- Be specific and actionable

Respond with just the one next step, no preamble."""
        messages = [Message(role="user", content=prompt)]
        return await self.client.chat(
            messages=messages,
            model="fast",
            temperature=0.3,
            max_tokens=150,
        )
