"""Pluggable routing system for adaptive agent selection.

This module provides explainable, deterministic routing with full traces.
Three router implementations:
- RouterV0Heuristic: Fast, rule-based, deterministic
- RouterV1LLM: LLM-powered selection with traces
- RouterV2Hybrid: Heuristic first, LLM for tie-breaks
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .openrouter import OpenRouterClient, Message


class RouterVersion(str, Enum):
    """Available router implementations."""
    HEURISTIC = "heuristic"  # V0: Fast, deterministic, rule-based
    LLM = "llm"              # V1: LLM-powered selection
    HYBRID = "hybrid"        # V2: Heuristic + LLM tie-break


@dataclass
class AgentScore:
    """Score and rationale for a single agent."""
    agent_name: str
    score: float  # 0.0 to 1.0
    rationale: str
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class RoutingTrace:
    """Complete trace of a routing decision for explainability."""
    router_version: RouterVersion
    user_input: str
    considered_agents: list[AgentScore]
    selected_agents: list[str]
    selection_rationale: str
    confidence: float  # Overall confidence 0.0 to 1.0
    tie_break_used: bool = False
    tie_break_method: str | None = None


class BaseRouter(ABC):
    """Abstract base class for agent routers."""

    @property
    @abstractmethod
    def version(self) -> RouterVersion:
        """Router version identifier."""
        pass

    @abstractmethod
    async def route(
        self,
        user_input: str,
        context: str | None = None,
        history: list[dict] | None = None,
        goals: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> RoutingTrace:
        """Select agents and return full routing trace."""
        pass


class RouterV0Heuristic(BaseRouter):
    """Fast, deterministic, rule-based router.

    Uses pattern matching to select agents:
    - Clarifier: vague quantifiers, undefined nouns, multiple goals
    - Advocate: strong claims, single-sided arguments, moral certainty
    - Synthesizer: multi-paragraph, multi-topic, after N turns
    - Expander: user stuck, asks "what should I do", narrow framing
    - Socratic: default when uncertain
    """

    # Pattern definitions for each agent
    PATTERNS = {
        "clarifier": {
            "vague_quantifiers": [
                r"\b(too much|too little|a lot|a few|some|many|better|worse|good|bad|normal|enough)\b",
                r"\b(should|ought|need to|have to)\b",
                r"\b(success|failure|effective|efficient|optimal)\b",
            ],
            "undefined_nouns": [
                r"\b(it|this|that|they|them|those|these)\b(?!\s+(is|are|was|were|will|would|could|should|can))",
                r"\b(the system|the thing|the stuff|the issue|the problem)\b",
            ],
            "multiple_goals": [
                r"\b(and also|but also|while also|at the same time)\b",
                r"\b(on one hand|on the other hand)\b",
            ],
            "vague_terms": [
                r"\b(fair|reasonable|appropriate|suitable|proper)\b",
                r"\b(kind of|sort of|somewhat|rather|pretty)\b",
            ],
        },
        "advocate": {
            "strong_claims": [
                r"\b(obviously|clearly|definitely|certainly|absolutely|undoubtedly)\b",
                r"\b(everyone knows|everyone agrees|it's clear that|it's obvious that)\b",
                r"\b(always|never|impossible|guaranteed|certain)\b",
            ],
            "single_sided": [
                r"\b(the only|the best|the worst|the right|the wrong)\b",
                r"\b(must be|has to be|can only be)\b",
            ],
            "moral_certainty": [
                r"\b(morally|ethically|fundamentally|inherently)\b",
                r"\b(evil|wrong|bad|immoral|unethical)\b.*\b(is|are)\b",
                r"\b(right thing|wrong thing|moral thing)\b",
            ],
        },
        "synthesizer": {
            "complexity": [
                r"(\n.*){3,}",  # Multiple paragraphs/lines
                r"\b(first|second|third|finally|additionally|moreover|furthermore)\b",
                r"\b(on one hand|however|but|although|yet|still)\b",
            ],
            "connections": [
                r"\b(relates to|connects to|similar to|different from|compared to)\b",
                r"\b(pattern|theme|thread|common|underlying)\b",
            ],
        },
        "expander": {
            "stuck": [
                r"\b(stuck|blocked|can't figure|don't know|not sure|confused|lost)\b",
                r"\b(what should i do|what do i do|how do i|help me)\b",
                r"\bi('m| am) (not sure|unsure|uncertain|confused)\b",
            ],
            "narrow_framing": [
                r"\b(only option|no choice|have to|must|can only)\b",
                r"\b(binary|either.+or|black.+white|all.+nothing)\b",
            ],
            "seeking_advice": [
                r"\?(.*should|.*would you|.*recommend|.*suggest|.*think)",
                r"\b(advice|guidance|direction|recommendation)\b",
            ],
        },
        "socratic": {
            "claims_to_examine": [
                r"\b(i think|i believe|i feel|in my opinion|my view)\b",
                r"\b(because|since|therefore|thus|hence|so)\b",
            ],
            "assumptions": [
                r"\b(assume|assuming|assumption|presume|suppose)\b",
                r"\b(given that|considering that|based on)\b",
            ],
        },
    }

    # Weights for pattern categories
    CATEGORY_WEIGHTS = {
        "clarifier": {"vague_quantifiers": 0.3, "undefined_nouns": 0.3, "multiple_goals": 0.2, "vague_terms": 0.2},
        "advocate": {"strong_claims": 0.4, "single_sided": 0.3, "moral_certainty": 0.3},
        "synthesizer": {"complexity": 0.5, "connections": 0.5},
        "expander": {"stuck": 0.4, "narrow_framing": 0.3, "seeking_advice": 0.3},
        "socratic": {"claims_to_examine": 0.5, "assumptions": 0.5},
    }

    @property
    def version(self) -> RouterVersion:
        return RouterVersion.HEURISTIC

    def _score_agent(self, user_input: str, agent_name: str) -> AgentScore:
        """Calculate score for a single agent based on pattern matching."""
        patterns = self.PATTERNS.get(agent_name, {})
        weights = self.CATEGORY_WEIGHTS.get(agent_name, {})

        total_score = 0.0
        matched_patterns = []
        rationale_parts = []

        input_lower = user_input.lower()

        for category, pattern_list in patterns.items():
            category_matched = False
            for pattern in pattern_list:
                matches = re.findall(pattern, input_lower, re.IGNORECASE)
                if matches:
                    category_matched = True
                    matched_patterns.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])

            if category_matched:
                weight = weights.get(category, 0.2)
                total_score += weight
                rationale_parts.append(f"{category.replace('_', ' ')}")

        # Normalize score to 0-1 range
        normalized_score = min(1.0, total_score)

        rationale = f"Matched: {', '.join(rationale_parts)}" if rationale_parts else "No specific patterns matched"

        return AgentScore(
            agent_name=agent_name,
            score=normalized_score,
            rationale=rationale,
            matched_patterns=list(set(matched_patterns))[:5],  # Top 5 unique matches
        )

    async def route(
        self,
        user_input: str,
        context: str | None = None,
        history: list[dict] | None = None,
        goals: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> RoutingTrace:
        """Route using heuristic pattern matching."""
        # Score all agents
        agent_names = ["clarifier", "advocate", "synthesizer", "expander", "socratic"]
        scores = [self._score_agent(user_input, name) for name in agent_names]

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)

        # Select top 2-3 agents with score > 0.1
        selected = []
        for score in scores:
            if score.score > 0.1 and len(selected) < 3:
                selected.append(score.agent_name)
            elif len(selected) >= 2:
                break

        # Always include at least socratic if we don't have 2
        if len(selected) < 2:
            if "socratic" not in selected:
                selected.append("socratic")
            # Add next best if still < 2
            for score in scores:
                if score.agent_name not in selected and len(selected) < 2:
                    selected.append(score.agent_name)
                    break

        # Calculate overall confidence
        avg_score = sum(s.score for s in scores if s.agent_name in selected) / len(selected) if selected else 0.0

        # Build selection rationale
        top_reasons = [f"{s.agent_name}: {s.rationale}" for s in scores if s.agent_name in selected]

        return RoutingTrace(
            router_version=self.version,
            user_input=user_input,
            considered_agents=scores,
            selected_agents=selected,
            selection_rationale="; ".join(top_reasons),
            confidence=avg_score,
        )


class RouterV1LLM(BaseRouter):
    """LLM-powered router with full traces.

    Uses a fast model to analyze input and select agents,
    while capturing the reasoning for explainability.
    """

    def __init__(self, client: "OpenRouterClient"):
        self.client = client

    @property
    def version(self) -> RouterVersion:
        return RouterVersion.LLM

    async def route(
        self,
        user_input: str,
        context: str | None = None,
        history: list[dict] | None = None,
        goals: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> RoutingTrace:
        """Route using LLM analysis."""
        from .openrouter import Message

        # Build context string
        context_parts = []
        if context:
            context_parts.append(f"Session context: {context}")
        if goals:
            context_parts.append(f"User goals: {', '.join(goals)}")
        if constraints:
            context_parts.append(f"Constraints: {', '.join(constraints)}")
        if history:
            turn_count = len([h for h in history if h.get("type") == "user"])
            context_parts.append(f"Conversation turn: {turn_count + 1}")

        context_str = "\n".join(context_parts) if context_parts else ""

        selection_prompt = f"""You are an orchestrator for a thinking assistance system. Analyze the input and select which agents should respond.

Available agents:
- socratic: Asks probing questions. Use when user makes claims or needs to examine beliefs.
- advocate: Challenges assumptions, presents counterarguments. Use when user seems certain.
- clarifier: Identifies ambiguity, asks for definitions. Use when terms are vague.
- synthesizer: Finds patterns, organizes ideas. Use for complexity or connecting ideas.
- expander: Offers alternative perspectives. Use when thinking seems narrow.

User input: {user_input}
{context_str}

Respond in this exact format:
SELECTED: agent1, agent2
SCORES: agent1=0.8, agent2=0.6, agent3=0.3
RATIONALE: Brief explanation of why each selected agent is appropriate.
CONFIDENCE: 0.X

Select 2-3 agents that would most help the user think deeper."""

        messages = [Message(role="user", content=selection_prompt)]

        response = await self.client.chat(
            messages=messages,
            model="fast",
            temperature=0.3,
            max_tokens=200,
        )

        # Parse response
        selected = []
        scores = []
        rationale = ""
        confidence = 0.5

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("SELECTED:"):
                agents_str = line.replace("SELECTED:", "").strip()
                selected = [a.strip().lower() for a in agents_str.split(",")]
                selected = [a for a in selected if a in ["socratic", "advocate", "clarifier", "synthesizer", "expander"]]
            elif line.startswith("SCORES:"):
                scores_str = line.replace("SCORES:", "").strip()
                for score_part in scores_str.split(","):
                    if "=" in score_part:
                        name, val = score_part.split("=")
                        try:
                            score_val = float(val.strip())
                            scores.append(AgentScore(
                                agent_name=name.strip().lower(),
                                score=score_val,
                                rationale="LLM-scored",
                            ))
                        except ValueError:
                            pass
            elif line.startswith("RATIONALE:"):
                rationale = line.replace("RATIONALE:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass

        # Ensure we have valid selection
        if not selected:
            selected = ["socratic", "clarifier"]

        # Fill in missing scores
        all_agents = ["socratic", "advocate", "clarifier", "synthesizer", "expander"]
        existing_names = {s.agent_name for s in scores}
        for agent in all_agents:
            if agent not in existing_names:
                scores.append(AgentScore(
                    agent_name=agent,
                    score=0.5 if agent in selected else 0.2,
                    rationale="Default score",
                ))

        return RoutingTrace(
            router_version=self.version,
            user_input=user_input,
            considered_agents=scores,
            selected_agents=selected,
            selection_rationale=rationale,
            confidence=confidence,
        )


class RouterV2Hybrid(BaseRouter):
    """Hybrid router: heuristic first, LLM for tie-breaks.

    Combines the speed of heuristics with LLM intelligence:
    1. Run heuristic scoring
    2. If clear winner (top score > 0.5), use heuristic
    3. If tie or low confidence, use LLM to decide
    """

    def __init__(self, client: "OpenRouterClient"):
        self.heuristic = RouterV0Heuristic()
        self.llm = RouterV1LLM(client)

    @property
    def version(self) -> RouterVersion:
        return RouterVersion.HYBRID

    async def route(
        self,
        user_input: str,
        context: str | None = None,
        history: list[dict] | None = None,
        goals: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> RoutingTrace:
        """Route using hybrid approach."""
        # First, get heuristic scores
        heuristic_trace = await self.heuristic.route(user_input, context, history, goals, constraints)

        # Check if heuristic is confident
        top_scores = sorted(heuristic_trace.considered_agents, key=lambda x: x.score, reverse=True)

        # If top score is high and there's clear separation, use heuristic
        if top_scores and top_scores[0].score > 0.5:
            if len(top_scores) < 2 or (top_scores[0].score - top_scores[1].score) > 0.2:
                # Clear winner, use heuristic
                return RoutingTrace(
                    router_version=self.version,
                    user_input=user_input,
                    considered_agents=heuristic_trace.considered_agents,
                    selected_agents=heuristic_trace.selected_agents,
                    selection_rationale=f"Heuristic (high confidence): {heuristic_trace.selection_rationale}",
                    confidence=heuristic_trace.confidence,
                    tie_break_used=False,
                )

        # Otherwise, use LLM for tie-break
        llm_trace = await self.llm.route(user_input, context, history, goals, constraints)

        # Merge scores (average of both)
        merged_scores = {}
        for score in heuristic_trace.considered_agents:
            merged_scores[score.agent_name] = {
                "heuristic": score.score,
                "llm": 0.0,
                "patterns": score.matched_patterns,
            }
        for score in llm_trace.considered_agents:
            if score.agent_name in merged_scores:
                merged_scores[score.agent_name]["llm"] = score.score
            else:
                merged_scores[score.agent_name] = {"heuristic": 0.0, "llm": score.score, "patterns": []}

        final_scores = []
        for name, data in merged_scores.items():
            avg = (data["heuristic"] + data["llm"]) / 2
            final_scores.append(AgentScore(
                agent_name=name,
                score=avg,
                rationale=f"Heuristic: {data['heuristic']:.2f}, LLM: {data['llm']:.2f}",
                matched_patterns=data.get("patterns", []),
            ))

        return RoutingTrace(
            router_version=self.version,
            user_input=user_input,
            considered_agents=final_scores,
            selected_agents=llm_trace.selected_agents,  # Use LLM's selection for tie-break
            selection_rationale=f"Hybrid (LLM tie-break): {llm_trace.selection_rationale}",
            confidence=(heuristic_trace.confidence + llm_trace.confidence) / 2,
            tie_break_used=True,
            tie_break_method="LLM",
        )


def create_router(version: RouterVersion, client: "OpenRouterClient | None" = None) -> BaseRouter:
    """Factory function to create a router instance.

    Args:
        version: Which router version to use
        client: OpenRouter client (required for LLM and Hybrid routers)

    Returns:
        Router instance
    """
    if version == RouterVersion.HEURISTIC:
        return RouterV0Heuristic()
    elif version == RouterVersion.LLM:
        if not client:
            raise ValueError("LLM router requires OpenRouterClient")
        return RouterV1LLM(client)
    elif version == RouterVersion.HYBRID:
        if not client:
            raise ValueError("Hybrid router requires OpenRouterClient")
        return RouterV2Hybrid(client)
    else:
        raise ValueError(f"Unknown router version: {version}")
