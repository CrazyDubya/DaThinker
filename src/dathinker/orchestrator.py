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
from .router import (
    BaseRouter,
    RouterVersion,
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
    OPEN = "open"          # Not yet validated
    CONFIRMED = "confirmed"  # Validated as true
    CONTESTED = "contested"  # Challenged by agents
    REVISED = "revised"     # Modified from original


class SynthesisStyle(str, Enum):
    """Output style for synthesis."""
    MEMO = "memo"       # Formal memo format
    OUTLINE = "outline"  # Hierarchical outline
    DEBATE = "debate"    # Pro/con debate format
    TODO = "todo"        # Action-oriented todo list


@dataclass
class PinnedStatement:
    """A user-pinned statement or assumption."""
    id: str
    content: str
    status: AssumptionStatus = AssumptionStatus.OPEN
    turn_created: int = 0
    turn_last_referenced: int = 0
    agent_references: list[str] = field(default_factory=list)  # Which agents referenced it


@dataclass
class Goal:
    """A session goal the user is optimizing for."""
    id: str
    content: str
    priority: int = 1  # 1 = highest
    active: bool = True


@dataclass
class Constraint:
    """A constraint agents must respect."""
    id: str
    content: str
    hard: bool = True  # Hard = must respect, Soft = prefer to respect


@dataclass
class MultiLevelSynthesis:
    """Multi-level synthesis output."""
    tldr: list[str]  # 5 bullet TL;DR
    key_claims: list[str]
    evidence: list[str]
    assumptions: list[str]
    open_questions: list[str]
    conflicts: list[str]  # Conflicts between agents
    next_moves: list[str]  # Experiments, decisions, questions
    raw_text: str  # Full synthesis text


@dataclass
class ThinkingSession:
    """A thinking session with history and context."""
    id: str
    topic: str
    mode: ThinkingMode
    history: list[dict] = field(default_factory=list)  # Full conversation history
    insights: list[str] = field(default_factory=list)  # Accumulated insights
    questions: list[str] = field(default_factory=list)  # Open questions

    # New: Conversation control surfaces
    pins: list[PinnedStatement] = field(default_factory=list)
    goals: list[Goal] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)

    # New: Routing traces for explainability
    routing_traces: list[RoutingTrace] = field(default_factory=list)

    # Counters
    _pin_counter: int = 0
    _goal_counter: int = 0
    _constraint_counter: int = 0


class ThinkingOrchestrator:
    """Orchestrates multiple thinking agents to help users think deeper.

    The orchestrator:
    - Manages a panel of thinking agents
    - Coordinates their interactions based on the thinking mode
    - Maintains session context and history
    - Synthesizes agent outputs into coherent dialogue
    - Provides explainable adaptive routing
    """

    def __init__(
        self,
        client: OpenRouterClient | None = None,
        model: str = "balanced",
        router_version: RouterVersion = RouterVersion.HYBRID,
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

        # Initialize router
        self.router = create_router(router_version, self.client)
        self.router_version = router_version

        self.active_session: ThinkingSession | None = None
        self._session_counter = 0

    def set_router(self, version: RouterVersion) -> None:
        """Change the routing strategy."""
        self.router = create_router(version, self.client)
        self.router_version = version

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

    # ========== Conversation Control Surfaces ==========

    def pin(self, content: str) -> PinnedStatement:
        """Pin a statement as a working assumption."""
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
        # Sort by priority
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
        """Mark a goal as inactive (completed or abandoned)."""
        if not self.active_session:
            return False

        for goal in self.active_session.goals:
            if goal.id == goal_id:
                goal.active = False
                return True
        return False

    def add_constraint(self, content: str, hard: bool = True) -> Constraint:
        """Add a constraint agents must respect."""
        if not self.active_session:
            raise ValueError("No active session")

        self.active_session._constraint_counter += 1
        constraint = Constraint(
            id=f"constraint_{self.active_session._constraint_counter}",
            content=content,
            hard=hard,
        )
        self.active_session.constraints.append(constraint)
        return constraint

    def get_constraints(self) -> list[Constraint]:
        """Get all constraints."""
        if not self.active_session:
            return []
        return self.active_session.constraints

    def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint."""
        if not self.active_session:
            return False

        for i, c in enumerate(self.active_session.constraints):
            if c.id == constraint_id:
                self.active_session.constraints.pop(i)
                return True
        return False

    def _build_context_for_agents(self) -> str:
        """Build context string including goals, constraints, and pins."""
        if not self.active_session:
            return ""

        parts = []

        goals = self.get_goals(active_only=True)
        if goals:
            goal_strs = [f"- {g.content}" for g in goals]
            parts.append("SESSION GOALS:\n" + "\n".join(goal_strs))

        constraints = self.get_constraints()
        if constraints:
            constraint_strs = [f"- {'[HARD]' if c.hard else '[SOFT]'} {c.content}" for c in constraints]
            parts.append("CONSTRAINTS:\n" + "\n".join(constraint_strs))

        pins = self.get_pins()
        if pins:
            pin_strs = [f"- [{p.status.value}] {p.content}" for p in pins]
            parts.append("WORKING ASSUMPTIONS:\n" + "\n".join(pin_strs))

        return "\n\n".join(parts)

    # ========== Core Thinking Methods ==========

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

        # Build full context
        session_context = self._build_context_for_agents()
        full_context = f"{session_context}\n\n{context}" if context else session_context

        response = await agent.think(user_input, full_context if full_context else None)

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

        # Build full context
        session_context = self._build_context_for_agents()
        full_context = f"{session_context}\n\n{context}" if context else session_context

        # Run all agents in parallel
        tasks = [
            self.agents[name].think(user_input, full_context if full_context else None)
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
    ) -> tuple[list[AgentResponse], RoutingTrace]:
        """Adaptively choose and sequence agents based on the input.

        Returns both the responses AND the routing trace for explainability.
        """
        # Build full context
        session_context = self._build_context_for_agents()
        full_context = f"{session_context}\n\n{context}" if context else session_context

        # Get routing decision with trace
        routing_trace = await self.router.route(
            user_input=user_input,
            context=full_context,
            history=self.active_session.history if self.active_session else None,
            goals=[g.content for g in self.get_goals()],
            constraints=[c.content for c in self.get_constraints()],
        )

        # Store trace in session
        if self.active_session:
            self.active_session.routing_traces.append(routing_trace)

        responses: list[AgentResponse] = []

        # Get responses in the suggested order
        for agent_name in routing_trace.selected_agents:
            agent = self.agents.get(agent_name)
            if not agent:
                continue

            response = await agent.think(
                user_input,
                full_context,
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
                "routing_trace": routing_trace,  # Include trace in history
            })
            for response in responses:
                self.active_session.history.append({
                    "type": "agent",
                    "agent": response.agent_name,
                    "response": response,
                })
                self.active_session.insights.extend(response.insights)
                self.active_session.questions.extend(response.questions)

        return responses, routing_trace

    def get_last_routing_trace(self) -> RoutingTrace | None:
        """Get the most recent routing trace."""
        if self.active_session and self.active_session.routing_traces:
            return self.active_session.routing_traces[-1]
        return None

    # ========== Multi-Level Synthesis ==========

    async def synthesize_session(
        self,
        style: SynthesisStyle = SynthesisStyle.MEMO,
    ) -> MultiLevelSynthesis:
        """Generate a multi-level synthesis of the current thinking session.

        Returns three tiers:
        1. TL;DR (5 bullets)
        2. Map (claims, evidence, assumptions, questions, conflicts)
        3. Next moves (experiments, decisions, questions to ask)
        """
        if not self.active_session or not self.active_session.history:
            return MultiLevelSynthesis(
                tldr=["No active session or empty history."],
                key_claims=[],
                evidence=[],
                assumptions=[],
                open_questions=[],
                conflicts=[],
                next_moves=[],
                raw_text="No active session or empty history.",
            )

        # Build context from session
        history_text = []
        for entry in self.active_session.history:
            if entry["type"] == "user":
                history_text.append(f"USER: {entry['content']}")
            else:
                history_text.append(f"{entry['agent'].upper()}: {entry['response'].content}")

        # Include pins, goals, constraints
        context_parts = []
        if self.active_session.pins:
            pins_text = "\n".join([f"- [{p.status.value}] {p.content}" for p in self.active_session.pins])
            context_parts.append(f"Working Assumptions:\n{pins_text}")
        if self.active_session.goals:
            goals_text = "\n".join([f"- {g.content}" for g in self.active_session.goals if g.active])
            context_parts.append(f"Goals:\n{goals_text}")

        style_instructions = {
            SynthesisStyle.MEMO: "Format as a professional memo with clear sections.",
            SynthesisStyle.OUTLINE: "Format as a hierarchical outline with indentation.",
            SynthesisStyle.DEBATE: "Format as a debate showing pro/con for key points.",
            SynthesisStyle.TODO: "Format as an action-oriented todo list with priorities.",
        }

        synthesis_prompt = f"""Review this thinking session and provide a multi-level synthesis.

Topic: {self.active_session.topic}
{chr(10).join(context_parts) if context_parts else ''}

Session history:
{chr(10).join(history_text)}

Provide synthesis in this EXACT format (use these exact headers):

## TL;DR
- [5 bullet points summarizing the key takeaways]

## KEY CLAIMS
- [Claims that emerged or were examined]

## EVIDENCE
- [Evidence or reasoning provided]

## ASSUMPTIONS
- [Underlying assumptions, stated or implied]

## OPEN QUESTIONS
- [Questions that remain unanswered]

## CONFLICTS
- [Areas where agents disagreed or tensions exist]

## NEXT MOVES
- [Concrete next steps: experiments to run, decisions to make, questions to ask someone]

{style_instructions.get(style, '')}

Be concise but substantive. Focus on helping the user see their progress and what remains."""

        messages = [Message(role="user", content=synthesis_prompt)]

        raw_text = await self.client.chat(
            messages=messages,
            model=self.model,
            temperature=0.5,
        )

        # Parse the structured response
        return self._parse_synthesis(raw_text)

    def _parse_synthesis(self, raw_text: str) -> MultiLevelSynthesis:
        """Parse synthesis response into structured format."""
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

            # Check for section headers
            if line.startswith("##") or line.startswith("**"):
                header = line.replace("#", "").replace("*", "").strip().lower()
                for key, section in section_map.items():
                    if key in header:
                        current_section = section
                        break

            # Extract bullet points
            elif line.startswith(("-", "*", "•")) and current_section:
                content = line.lstrip("-*• ").strip()
                if content:
                    sections[current_section].append(content)

            # Also capture numbered items
            elif line and line[0].isdigit() and current_section:
                import re
                match = re.match(r'^\d+[\.\)]\s*(.+)', line)
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
        """North star: What is the smallest next step that would reduce uncertainty the most?"""
        if not self.active_session or not self.active_session.history:
            return "Start by sharing what you're thinking about."

        # Build context
        history_text = []
        for entry in self.active_session.history[-10:]:  # Last 10 entries
            if entry["type"] == "user":
                history_text.append(f"USER: {entry['content']}")
            else:
                history_text.append(f"{entry['agent'].upper()}: {entry['response'].content}")

        questions = list(set(self.active_session.questions))[-5:]  # Last 5 unique questions
        assumptions = [p.content for p in self.active_session.pins if p.status == AssumptionStatus.OPEN]

        prompt = f"""Based on this thinking session, identify the SINGLE smallest next step that would most reduce uncertainty.

Topic: {self.active_session.topic}

Recent conversation:
{chr(10).join(history_text)}

Open questions: {questions if questions else 'None identified yet'}
Untested assumptions: {assumptions if assumptions else 'None pinned yet'}

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

    def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        if not self.active_session:
            return {"error": "No active session"}

        return {
            "id": self.active_session.id,
            "topic": self.active_session.topic,
            "mode": self.active_session.mode.value,
            "router": self.router_version.value,
            "turns": len([h for h in self.active_session.history if h["type"] == "user"]),
            "unique_insights": len(set(self.active_session.insights)),
            "open_questions": len(set(self.active_session.questions)),
            "pins": len(self.active_session.pins),
            "goals": len([g for g in self.active_session.goals if g.active]),
            "constraints": len(self.active_session.constraints),
            "routing_traces": len(self.active_session.routing_traces),
        }
