"""
Pluggable routing system for adaptive agent selection.

Provides three router implementations:
- RouterV0Heuristic: Fast, deterministic pattern-based routing
- RouterV1LLM: Uses LLM for intelligent routing decisions
- RouterV2Hybrid: Heuristic first, LLM tie-break

Each router produces a RoutingTrace for full explainability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Optional

from .agents.base import AgentRole


class RouterType(str, Enum):
    """Available router implementations."""
    HEURISTIC = "heuristic"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class AgentScore:
    """Score and reasoning for a single agent."""
    agent_name: str
    role: AgentRole
    score: float  # 0.0 to 1.0
    reasons: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)

    @property
    def selected(self) -> bool:
        """Whether this agent was selected (score >= threshold)."""
        return self.score >= 0.5


@dataclass
class RoutingTrace:
    """
    Complete trace of routing decision for explainability.

    This is returned with every adaptive mode response to make
    the routing decision fully transparent and debuggable.
    """
    router_type: RouterType
    input_summary: str  # First 100 chars of input
    agent_scores: list[AgentScore]
    selected_agents: list[str]
    selection_order: list[str]  # Order agents will respond
    confidence: float  # Overall confidence in selection (0.0-1.0)
    reasoning: str  # Human-readable explanation
    fallback_used: bool = False  # True if fell back to default
    llm_override: bool = False  # True if LLM changed heuristic decision

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "router_type": self.router_type.value,
            "input_summary": self.input_summary,
            "agent_scores": [
                {
                    "agent": s.agent_name,
                    "role": s.role.value,
                    "score": round(s.score, 2),
                    "reasons": s.reasons,
                    "patterns": s.matched_patterns,
                    "selected": s.selected,
                }
                for s in self.agent_scores
            ],
            "selected_agents": self.selected_agents,
            "selection_order": self.selection_order,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "fallback_used": self.fallback_used,
            "llm_override": self.llm_override,
        }

    def format_summary(self) -> str:
        """Format a concise summary for CLI display."""
        lines = [
            f"Router: {self.router_type.value.upper()}",
            f"Selected: {' -> '.join(self.selection_order)}",
            f"Confidence: {self.confidence:.0%}",
        ]
        if self.fallback_used:
            lines.append("(fallback to default)")
        if self.llm_override:
            lines.append("(LLM override applied)")
        return " | ".join(lines)

    def format_detailed(self) -> str:
        """Format detailed trace for verbose output."""
        lines = [
            "=" * 60,
            "ROUTING TRACE",
            "=" * 60,
            f"Router Type: {self.router_type.value}",
            f"Input: \"{self.input_summary}...\"" if len(self.input_summary) >= 100 else f"Input: \"{self.input_summary}\"",
            "",
            "Agent Scores:",
        ]

        for score in sorted(self.agent_scores, key=lambda x: x.score, reverse=True):
            status = "[SELECTED]" if score.selected else ""
            lines.append(f"  {score.agent_name} ({score.role.value}): {score.score:.2f} {status}")
            for reason in score.reasons:
                lines.append(f"    - {reason}")
            if score.matched_patterns:
                lines.append(f"    Patterns: {', '.join(score.matched_patterns[:3])}")

        lines.extend([
            "",
            f"Selection Order: {' -> '.join(self.selection_order)}",
            f"Confidence: {self.confidence:.0%}",
            "",
            f"Reasoning: {self.reasoning}",
            "=" * 60,
        ])

        return "\n".join(lines)


class BaseRouter(ABC):
    """Abstract base class for routing implementations."""

    @property
    @abstractmethod
    def router_type(self) -> RouterType:
        """Return the router type identifier."""
        pass

    @abstractmethod
    async def route(
        self,
        user_input: str,
        context: Optional[str] = None,
        session_history: Optional[list[dict]] = None,
        turn_count: int = 0,
    ) -> RoutingTrace:
        """
        Determine which agents should respond and in what order.

        Args:
            user_input: The user's current message
            context: Optional context from session
            session_history: List of previous turns in session
            turn_count: Number of turns so far

        Returns:
            RoutingTrace with full decision explanation
        """
        pass


class RouterV0Heuristic(BaseRouter):
    """
    Fast, deterministic pattern-based router.

    Uses regex patterns and linguistic markers to select agents.
    Surprisingly effective for most use cases.
    """

    # Vague quantifiers that trigger Clarifier
    VAGUE_QUANTIFIERS = [
        r"\btoo much\b", r"\btoo little\b", r"\bbetter\b", r"\bworse\b",
        r"\bnormal\b", r"\benough\b", r"\ba lot\b", r"\bsome\b",
        r"\bmany\b", r"\bfew\b", r"\bmost\b", r"\bkind of\b",
        r"\bsort of\b", r"\bpretty\b", r"\bquite\b", r"\breally\b",
        r"\bvery\b", r"\bsomewhat\b", r"\bfairly\b",
    ]

    # Undefined references that trigger Clarifier
    UNDEFINED_REFS = [
        r"\bit\b(?!\s+is\s+(?:not\s+)?(?:clear|obvious|evident))",
        r"\bthis\b(?!\s+is)", r"\bthat\b(?!\s+is)",
        r"\bthe system\b", r"\bthe thing\b", r"\bthe stuff\b",
        r"\bthey\b(?!\s+(?:are|were|will|have|had))",
        r"\bthem\b", r"\bthose\b(?!\s+(?:are|were))",
    ]

    # Strong claim language that triggers Advocate
    STRONG_CLAIMS = [
        r"\bobviously\b", r"\bclearly\b", r"\beveryone knows\b",
        r"\bno one\b", r"\balways\b", r"\bnever\b",
        r"\bwithout doubt\b", r"\bundoubtedly\b", r"\bcertainly\b",
        r"\bdefinitely\b", r"\babsolutely\b", r"\bimpossible\b",
        r"\bmust be\b", r"\bhas to be\b", r"\bonly way\b",
        r"\bthe best\b", r"\bthe worst\b", r"\bthe only\b",
        r"\bperfect\b", r"\bcompletely\b", r"\btotally\b",
    ]

    # Single-sided argument markers
    SINGLE_SIDED = [
        r"\bbecause\b.*\bso\b", r"\btherefore\b",
        r"\bthus\b", r"\bhence\b", r"\bconsequently\b",
        r"\bas a result\b", r"\bit follows\b",
        r"\bproves\b", r"\bdemonstrates\b", r"\bshows that\b",
    ]

    # Moral certainty markers
    MORAL_CERTAINTY = [
        r"\bwrong\b", r"\bright\b(?!\s+(?:now|away|here))",
        r"\bshould\b", r"\bmust\b(?!\s+be)",
        r"\bimmoral\b", r"\bunethical\b", r"\bevil\b",
        r"\bgood\b(?!\s+(?:at|for|with))", r"\bbad\b(?!\s+(?:at|for|with))",
        r"\bfair\b", r"\bunfair\b", r"\bjust\b", r"\bunjust\b",
    ]

    # Stuck/narrow framing that triggers Expander
    STUCK_MARKERS = [
        r"\bwhat should I do\b", r"\bwhat do I do\b",
        r"\bI don't know\b", r"\bI'm stuck\b", r"\bI'm confused\b",
        r"\bno idea\b", r"\bcan't decide\b", r"\bcan't figure\b",
        r"\bhelp me\b", r"\bi need help\b", r"\bi need advice\b",
        r"\bwhat would you\b", r"\bwhat do you think\b",
        r"\beither.*or\b", r"\bonly two\b", r"\bonly option\b",
    ]

    # Narrow framing markers
    NARROW_FRAMING = [
        r"\bthe only\b", r"\bjust\b(?!\s+(?:now|then|like))",
        r"\bsimply\b", r"\bmerely\b", r"\bnothing but\b",
        r"\bno other\b", r"\bno choice\b", r"\bhave to\b",
    ]

    # Multi-topic/synthesis markers
    SYNTHESIS_MARKERS = [
        r"\bon one hand\b.*\bon the other\b",
        r"\bfirstly\b.*\bsecondly\b", r"\bfirst\b.*\bsecond\b.*\bthird\b",
        r"\bhowever\b", r"\bbut\b.*\balso\b", r"\bwhile\b.*\balso\b",
        r"\balthough\b", r"\bdespite\b", r"\bnevertheless\b",
        r"\bconflicting\b", r"\bcontradictory\b", r"\btension\b",
    ]

    # Question markers (Socrates as default)
    QUESTION_MARKERS = [
        r"\?$", r"\bwhy\b", r"\bhow\b(?!\s+(?:much|many))",
        r"\bwhat if\b", r"\bwhat about\b", r"\bdo you think\b",
    ]

    @property
    def router_type(self) -> RouterType:
        return RouterType.HEURISTIC

    def _count_patterns(self, text: str, patterns: list[str]) -> tuple[int, list[str]]:
        """Count pattern matches and return matched patterns."""
        text_lower = text.lower()
        count = 0
        matched = []
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                count += 1
                matched.append(pattern.replace(r"\b", "").replace("\\", ""))
        return count, matched

    def _calculate_scores(
        self,
        user_input: str,
        turn_count: int,
    ) -> dict[str, AgentScore]:
        """Calculate scores for each agent based on input patterns."""
        scores = {}

        # Clarifier scoring - increased weights for better detection
        vague_count, vague_patterns = self._count_patterns(user_input, self.VAGUE_QUANTIFIERS)
        ref_count, ref_patterns = self._count_patterns(user_input, self.UNDEFINED_REFS)

        clarifier_score = 0.0
        clarifier_reasons = []
        clarifier_patterns = []

        if vague_count > 0:
            # Increased weight: even 2 vague terms should trigger clarifier
            clarifier_score += min(vague_count * 0.3, 0.7)
            clarifier_reasons.append(f"Found {vague_count} vague quantifier(s)")
            clarifier_patterns.extend(vague_patterns)
        if ref_count > 0:
            # Increased weight for undefined references
            clarifier_score += min(ref_count * 0.25, 0.5)
            clarifier_reasons.append(f"Found {ref_count} undefined reference(s)")
            clarifier_patterns.extend(ref_patterns)

        # Multiple goals detection - check for list patterns
        # Lists with commas or "and" often indicate multiple goals needing clarification
        and_count = user_input.lower().count(" and ")
        comma_count = user_input.count(",")
        if and_count >= 3 or comma_count >= 4:
            clarifier_score += 0.6
            clarifier_reasons.append("Multiple goals or items detected")
        elif comma_count >= 3:
            # Three commas usually means a list of items
            clarifier_score += 0.5
            clarifier_reasons.append("Multiple items in list format detected")
        elif and_count >= 2 or comma_count >= 2:
            clarifier_score += 0.4
            clarifier_reasons.append("Several items that may need clarification")

        scores["clarifier"] = AgentScore(
            agent_name="Clarifier",
            role=AgentRole.CLARIFIER,
            score=min(clarifier_score, 1.0),
            reasons=clarifier_reasons,
            matched_patterns=clarifier_patterns[:5],
        )

        # Advocate scoring
        claim_count, claim_patterns = self._count_patterns(user_input, self.STRONG_CLAIMS)
        sided_count, sided_patterns = self._count_patterns(user_input, self.SINGLE_SIDED)
        moral_count, moral_patterns = self._count_patterns(user_input, self.MORAL_CERTAINTY)

        advocate_score = 0.0
        advocate_reasons = []
        advocate_patterns = []

        if claim_count > 0:
            advocate_score += min(claim_count * 0.25, 0.6)
            advocate_reasons.append(f"Found {claim_count} strong claim marker(s)")
            advocate_patterns.extend(claim_patterns)
        if sided_count > 0:
            advocate_score += min(sided_count * 0.2, 0.3)
            advocate_reasons.append(f"Single-sided argument structure detected")
            advocate_patterns.extend(sided_patterns)
        if moral_count > 0:
            advocate_score += min(moral_count * 0.15, 0.3)
            advocate_reasons.append(f"Moral certainty language detected")
            advocate_patterns.extend(moral_patterns)

        scores["advocate"] = AgentScore(
            agent_name="Advocate",
            role=AgentRole.DEVILS_ADVOCATE,
            score=min(advocate_score, 1.0),
            reasons=advocate_reasons,
            matched_patterns=advocate_patterns[:5],
        )

        # Synthesizer scoring - increased weights
        synth_count, synth_patterns = self._count_patterns(user_input, self.SYNTHESIS_MARKERS)

        synthesizer_score = 0.0
        synthesizer_reasons = []
        synthesizer_patterns = []

        if synth_count > 0:
            # Even one synthesis marker should strongly boost
            synthesizer_score += min(synth_count * 0.5, 0.8)
            synthesizer_reasons.append(f"Found {synth_count} multi-perspective marker(s)")
            synthesizer_patterns.extend(synth_patterns)

        # Multi-paragraph detection
        paragraphs = len([p for p in user_input.split("\n\n") if p.strip()])
        if paragraphs >= 2:
            synthesizer_score += 0.4
            synthesizer_reasons.append(f"Multi-paragraph input ({paragraphs} paragraphs)")

        # Long input - tiered thresholds (long input = complexity = needs synthesis)
        input_len = len(user_input)
        if input_len > 500:
            synthesizer_score += 0.6  # Very long - strong synthesis signal
            synthesizer_reasons.append("Very long input needs organization")
        elif input_len > 300:
            synthesizer_score += 0.4
            synthesizer_reasons.append("Long input needs organization")

        # Periodic synthesis after N turns
        if turn_count > 0 and turn_count % 5 == 0:
            synthesizer_score += 0.4
            synthesizer_reasons.append(f"Periodic synthesis (turn {turn_count})")

        scores["synthesizer"] = AgentScore(
            agent_name="Synthesizer",
            role=AgentRole.SYNTHESIZER,
            score=min(synthesizer_score, 1.0),
            reasons=synthesizer_reasons,
            matched_patterns=synthesizer_patterns[:5],
        )

        # Expander scoring - increased weights
        stuck_count, stuck_patterns = self._count_patterns(user_input, self.STUCK_MARKERS)
        narrow_count, narrow_patterns = self._count_patterns(user_input, self.NARROW_FRAMING)

        expander_score = 0.0
        expander_reasons = []
        expander_patterns = []

        if stuck_count > 0:
            # Even one stuck marker should trigger expander
            expander_score += min(stuck_count * 0.5, 0.8)
            expander_reasons.append(f"User appears stuck or seeking direction")
            expander_patterns.extend(stuck_patterns)
        if narrow_count > 0:
            expander_score += min(narrow_count * 0.3, 0.5)
            expander_reasons.append(f"Narrow framing detected")
            expander_patterns.extend(narrow_patterns)

        scores["expander"] = AgentScore(
            agent_name="Expander",
            role=AgentRole.PERSPECTIVE,
            score=min(expander_score, 1.0),
            reasons=expander_reasons,
            matched_patterns=expander_patterns[:5],
        )

        # Socratic scoring (default + question markers)
        question_count, question_patterns = self._count_patterns(user_input, self.QUESTION_MARKERS)

        socratic_score = 0.25  # Lower base score to let specific agents win
        socratic_reasons = ["Default: keeps the conversation moving"]
        socratic_patterns = []

        if question_count > 0:
            socratic_score += min(question_count * 0.25, 0.5)
            socratic_reasons.append(f"User asking questions - engage with deeper inquiry")
            socratic_patterns.extend(question_patterns)

        # Only boost Socratic if no other agent scores at all
        max_other_score = max(
            scores["clarifier"].score,
            scores["advocate"].score,
            scores["synthesizer"].score,
            scores["expander"].score,
        )
        if max_other_score < 0.3:
            socratic_score += 0.35
            socratic_reasons.append("No strong signals for other agents")

        scores["socratic"] = AgentScore(
            agent_name="Socrates",
            role=AgentRole.SOCRATIC,
            score=min(socratic_score, 1.0),
            reasons=socratic_reasons,
            matched_patterns=socratic_patterns[:5],
        )

        return scores

    async def route(
        self,
        user_input: str,
        context: Optional[str] = None,
        session_history: Optional[list[dict]] = None,
        turn_count: int = 0,
    ) -> RoutingTrace:
        """Route using pattern-based heuristics."""
        scores = self._calculate_scores(user_input, turn_count)

        # Select agents with score >= 0.5, sorted by score
        selected = [
            name for name, score in sorted(
                scores.items(),
                key=lambda x: x[1].score,
                reverse=True,
            )
            if score.score >= 0.5
        ]

        # Ensure at least one agent (default to Socratic)
        fallback_used = False
        if not selected:
            selected = ["socratic"]
            fallback_used = True

        # Limit to 3 agents max
        selected = selected[:3]

        # Calculate confidence
        selected_scores = [scores[name].score for name in selected]
        confidence = sum(selected_scores) / len(selected_scores) if selected_scores else 0.5

        # Generate reasoning
        reasoning_parts = []
        for name in selected:
            score = scores[name]
            if score.reasons:
                reasoning_parts.append(f"{score.agent_name}: {score.reasons[0]}")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Default selection"

        return RoutingTrace(
            router_type=self.router_type,
            input_summary=user_input[:100],
            agent_scores=list(scores.values()),
            selected_agents=selected,
            selection_order=selected,
            confidence=confidence,
            reasoning=reasoning,
            fallback_used=fallback_used,
            llm_override=False,
        )


class RouterV1LLM(BaseRouter):
    """
    LLM-based router using fast model for intelligent decisions.

    More flexible than heuristics but slower and costs tokens.
    """

    def __init__(self, openrouter_client):
        """Initialize with OpenRouter client."""
        self.client = openrouter_client

    @property
    def router_type(self) -> RouterType:
        return RouterType.LLM

    async def route(
        self,
        user_input: str,
        context: Optional[str] = None,
        session_history: Optional[list[dict]] = None,
        turn_count: int = 0,
    ) -> RoutingTrace:
        """Route using LLM analysis."""
        from .openrouter import Message

        selection_prompt = f"""You are a routing system for a multi-agent thinking assistant.
Given the user's input, select 2-3 agents that should respond, in optimal order.

Available agents:
- socratic: Asks probing questions to deepen understanding. Use for claims that need examination.
- advocate: Challenges assumptions and plays devil's advocate. Use when user seems certain or one-sided.
- clarifier: Identifies vague language and asks for definitions. Use when terms are unclear.
- synthesizer: Finds patterns and organizes complex ideas. Use for multi-topic or long inputs.
- expander: Offers alternative perspectives and reframes. Use when user seems stuck or narrow.

User input: "{user_input[:500]}"
{"Context: " + context[:200] if context else ""}
Turn count: {turn_count}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
    "agents": ["agent1", "agent2"],
    "scores": {{"agent1": 0.8, "agent2": 0.6, ...}},
    "reasoning": "Brief explanation"
}}"""

        messages = [Message(role="user", content=selection_prompt)]

        try:
            response = await self.client.chat(messages, model="fast", temperature=0.3)

            # Parse JSON response
            import json
            # Clean response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)

            agents = result.get("agents", ["socratic"])
            score_dict = result.get("scores", {})
            reasoning = result.get("reasoning", "LLM selection")

            # Build agent scores
            agent_scores = []
            agent_names = {
                "socratic": ("Socrates", AgentRole.SOCRATIC),
                "advocate": ("Advocate", AgentRole.DEVILS_ADVOCATE),
                "clarifier": ("Clarifier", AgentRole.CLARIFIER),
                "synthesizer": ("Synthesizer", AgentRole.SYNTHESIZER),
                "expander": ("Expander", AgentRole.PERSPECTIVE),
            }

            for name, (display_name, role) in agent_names.items():
                score = score_dict.get(name, 0.3)
                agent_scores.append(AgentScore(
                    agent_name=display_name,
                    role=role,
                    score=score,
                    reasons=["LLM analysis"] if name in agents else [],
                ))

            confidence = sum(score_dict.get(a, 0.5) for a in agents) / len(agents) if agents else 0.5

            return RoutingTrace(
                router_type=self.router_type,
                input_summary=user_input[:100],
                agent_scores=agent_scores,
                selected_agents=agents,
                selection_order=agents,
                confidence=confidence,
                reasoning=reasoning,
                fallback_used=False,
                llm_override=False,
            )

        except Exception as e:
            # Fallback to Socratic on error
            return RoutingTrace(
                router_type=self.router_type,
                input_summary=user_input[:100],
                agent_scores=[
                    AgentScore("Socrates", AgentRole.SOCRATIC, 0.5, ["Fallback due to error"])
                ],
                selected_agents=["socratic"],
                selection_order=["socratic"],
                confidence=0.5,
                reasoning=f"Fallback due to LLM error: {str(e)[:50]}",
                fallback_used=True,
                llm_override=False,
            )


class RouterV2Hybrid(BaseRouter):
    """
    Hybrid router: heuristic first, LLM tie-break.

    Best of both worlds: fast and deterministic when patterns are clear,
    intelligent when ambiguous.
    """

    def __init__(self, openrouter_client):
        """Initialize with OpenRouter client."""
        self.heuristic_router = RouterV0Heuristic()
        self.llm_router = RouterV1LLM(openrouter_client)
        self.client = openrouter_client

    @property
    def router_type(self) -> RouterType:
        return RouterType.HYBRID

    async def route(
        self,
        user_input: str,
        context: Optional[str] = None,
        session_history: Optional[list[dict]] = None,
        turn_count: int = 0,
    ) -> RoutingTrace:
        """Route using heuristic with LLM tie-break."""
        # Get heuristic result
        heuristic_trace = await self.heuristic_router.route(
            user_input, context, session_history, turn_count
        )

        # If confidence is high, use heuristic result
        if heuristic_trace.confidence >= 0.7:
            return RoutingTrace(
                router_type=self.router_type,
                input_summary=heuristic_trace.input_summary,
                agent_scores=heuristic_trace.agent_scores,
                selected_agents=heuristic_trace.selected_agents,
                selection_order=heuristic_trace.selection_order,
                confidence=heuristic_trace.confidence,
                reasoning=f"Heuristic (high confidence): {heuristic_trace.reasoning}",
                fallback_used=heuristic_trace.fallback_used,
                llm_override=False,
            )

        # If confidence is low or fallback was used, consult LLM
        if heuristic_trace.confidence < 0.6 or heuristic_trace.fallback_used:
            llm_trace = await self.llm_router.route(
                user_input, context, session_history, turn_count
            )

            # Use LLM result if it has higher confidence
            if llm_trace.confidence > heuristic_trace.confidence:
                return RoutingTrace(
                    router_type=self.router_type,
                    input_summary=llm_trace.input_summary,
                    agent_scores=llm_trace.agent_scores,
                    selected_agents=llm_trace.selected_agents,
                    selection_order=llm_trace.selection_order,
                    confidence=llm_trace.confidence,
                    reasoning=f"LLM override: {llm_trace.reasoning}",
                    fallback_used=False,
                    llm_override=True,
                )

        # Otherwise use heuristic
        return RoutingTrace(
            router_type=self.router_type,
            input_summary=heuristic_trace.input_summary,
            agent_scores=heuristic_trace.agent_scores,
            selected_agents=heuristic_trace.selected_agents,
            selection_order=heuristic_trace.selection_order,
            confidence=heuristic_trace.confidence,
            reasoning=f"Heuristic: {heuristic_trace.reasoning}",
            fallback_used=heuristic_trace.fallback_used,
            llm_override=False,
        )


def create_router(
    router_type: RouterType,
    openrouter_client=None,
) -> BaseRouter:
    """
    Factory function to create a router instance.

    Args:
        router_type: Which router implementation to use
        openrouter_client: Required for LLM and Hybrid routers

    Returns:
        Router instance
    """
    if router_type == RouterType.HEURISTIC:
        return RouterV0Heuristic()
    elif router_type == RouterType.LLM:
        if openrouter_client is None:
            raise ValueError("LLM router requires openrouter_client")
        return RouterV1LLM(openrouter_client)
    elif router_type == RouterType.HYBRID:
        if openrouter_client is None:
            raise ValueError("Hybrid router requires openrouter_client")
        return RouterV2Hybrid(openrouter_client)
    else:
        raise ValueError(f"Unknown router type: {router_type}")
