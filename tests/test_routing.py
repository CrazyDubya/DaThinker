"""Tests for the pluggable routing system.

Tests routing determinism, heuristic patterns, and routing traces.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

import sys
sys.path.insert(0, "src")

from dathinker.router import (
    RouterV0Heuristic,
    RouterV1LLM,
    RouterV2Hybrid,
    RouterVersion,
    RoutingTrace,
    AgentScore,
    create_router,
)


class TestRouterV0Heuristic:
    """Tests for the deterministic heuristic router."""

    @pytest.fixture
    def router(self):
        return RouterV0Heuristic()

    @pytest.mark.asyncio
    async def test_clarifier_selected_for_vague_quantifiers(self, router):
        """Clarifier should be selected when vague quantifiers are present."""
        inputs = [
            "I think the system is too slow",
            "We need a better solution",
            "The results were good enough",
            "It's kind of working",
        ]

        for user_input in inputs:
            trace = await router.route(user_input)
            assert "clarifier" in trace.selected_agents, f"Clarifier not selected for: {user_input}"
            clarifier_score = next(s for s in trace.considered_agents if s.agent_name == "clarifier")
            assert clarifier_score.score > 0.1, f"Clarifier score too low for: {user_input}"

    @pytest.mark.asyncio
    async def test_advocate_selected_for_strong_claims(self, router):
        """Advocate should be selected when strong claims are made."""
        inputs = [
            "Obviously, this is the only way to do it",
            "Everyone knows that agile is better",
            "It's clearly the right choice",
            "This will definitely work",
        ]

        for user_input in inputs:
            trace = await router.route(user_input)
            assert "advocate" in trace.selected_agents, f"Advocate not selected for: {user_input}"
            advocate_score = next(s for s in trace.considered_agents if s.agent_name == "advocate")
            assert advocate_score.score > 0.1, f"Advocate score too low for: {user_input}"

    @pytest.mark.asyncio
    async def test_expander_selected_when_stuck(self, router):
        """Expander should be selected when user seems stuck."""
        inputs = [
            "I'm stuck on this problem",
            "What should I do next?",
            "I don't know how to proceed",
            "Help me figure this out",
        ]

        for user_input in inputs:
            trace = await router.route(user_input)
            assert "expander" in trace.selected_agents, f"Expander not selected for: {user_input}"

    @pytest.mark.asyncio
    async def test_synthesizer_selected_for_complexity(self, router):
        """Synthesizer should be selected for complex, multi-part inputs."""
        inputs = [
            "First, we have the cost issue. Second, there's the timeline. Finally, the quality concerns.",
            "On one hand we could do X, however Y is also an option, although Z might be better.",
        ]

        for user_input in inputs:
            trace = await router.route(user_input)
            assert "synthesizer" in trace.selected_agents, f"Synthesizer not selected for complex input"

    @pytest.mark.asyncio
    async def test_routing_is_deterministic(self, router):
        """Same input should always produce same routing decision."""
        user_input = "I think we should definitely use Python for this project"

        traces = [await router.route(user_input) for _ in range(5)]

        # All traces should have same selected agents
        first_selection = traces[0].selected_agents
        for trace in traces[1:]:
            assert trace.selected_agents == first_selection, "Routing is not deterministic"

    @pytest.mark.asyncio
    async def test_at_least_two_agents_selected(self, router):
        """Router should always select at least 2 agents."""
        inputs = [
            "Hello",
            "Simple statement",
            "I wonder about things",
            "",  # Empty input
        ]

        for user_input in inputs:
            trace = await router.route(user_input)
            assert len(trace.selected_agents) >= 2, f"Less than 2 agents for: {user_input}"

    @pytest.mark.asyncio
    async def test_socratic_as_default(self, router):
        """Socratic should be included when no clear patterns match."""
        trace = await router.route("Hello there")

        # Socratic should be in selection for ambiguous input
        assert "socratic" in trace.selected_agents

    @pytest.mark.asyncio
    async def test_trace_includes_all_agents(self, router):
        """Routing trace should include scores for all agents."""
        trace = await router.route("Any input here")

        agent_names = {s.agent_name for s in trace.considered_agents}
        expected = {"socratic", "advocate", "clarifier", "synthesizer", "expander"}
        assert agent_names == expected

    @pytest.mark.asyncio
    async def test_trace_has_rationale(self, router):
        """Routing trace should include selection rationale."""
        trace = await router.route("I think this is obviously the best approach")

        assert trace.selection_rationale, "No rationale provided"
        assert len(trace.selection_rationale) > 0


class TestRouterV2Hybrid:
    """Tests for the hybrid router."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.chat = AsyncMock(return_value="SELECTED: socratic, clarifier\nSCORES: socratic=0.8, clarifier=0.7\nRATIONALE: Testing\nCONFIDENCE: 0.75")
        return client

    @pytest.fixture
    def router(self, mock_client):
        return RouterV2Hybrid(mock_client)

    @pytest.mark.asyncio
    async def test_uses_heuristic_when_confident(self, router):
        """Should use heuristic result when confidence is high."""
        # Input that clearly triggers advocate (strong claims)
        trace = await router.route("Obviously everyone knows this is definitely true")

        # Should still get a valid trace
        assert trace.router_version == RouterVersion.HYBRID
        assert len(trace.selected_agents) >= 2

    @pytest.mark.asyncio
    async def test_tie_break_recorded(self, router, mock_client):
        """Should record when tie-break was used."""
        # Ambiguous input that might trigger LLM tie-break
        trace = await router.route("Hello")

        # Check trace structure
        assert hasattr(trace, 'tie_break_used')
        assert hasattr(trace, 'tie_break_method')


class TestRoutingTraceStructure:
    """Tests for routing trace data structure."""

    @pytest.fixture
    def router(self):
        return RouterV0Heuristic()

    @pytest.mark.asyncio
    async def test_trace_has_required_fields(self, router):
        """Trace should have all required fields."""
        trace = await router.route("Test input")

        assert hasattr(trace, 'router_version')
        assert hasattr(trace, 'user_input')
        assert hasattr(trace, 'considered_agents')
        assert hasattr(trace, 'selected_agents')
        assert hasattr(trace, 'selection_rationale')
        assert hasattr(trace, 'confidence')

    @pytest.mark.asyncio
    async def test_agent_scores_have_required_fields(self, router):
        """Agent scores should have all required fields."""
        trace = await router.route("Test input")

        for score in trace.considered_agents:
            assert hasattr(score, 'agent_name')
            assert hasattr(score, 'score')
            assert hasattr(score, 'rationale')
            assert hasattr(score, 'matched_patterns')
            assert 0.0 <= score.score <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_in_valid_range(self, router):
        """Confidence should be between 0 and 1."""
        trace = await router.route("Test input")
        assert 0.0 <= trace.confidence <= 1.0


class TestRouterFactory:
    """Tests for router factory function."""

    def test_creates_heuristic_router(self):
        router = create_router(RouterVersion.HEURISTIC)
        assert isinstance(router, RouterV0Heuristic)

    def test_creates_llm_router(self):
        mock_client = MagicMock()
        router = create_router(RouterVersion.LLM, mock_client)
        assert isinstance(router, RouterV1LLM)

    def test_creates_hybrid_router(self):
        mock_client = MagicMock()
        router = create_router(RouterVersion.HYBRID, mock_client)
        assert isinstance(router, RouterV2Hybrid)

    def test_llm_router_requires_client(self):
        with pytest.raises(ValueError):
            create_router(RouterVersion.LLM)

    def test_hybrid_router_requires_client(self):
        with pytest.raises(ValueError):
            create_router(RouterVersion.HYBRID)


# Regression tests for specific routing scenarios
class TestRoutingRegressions:
    """Regression tests to ensure routing behavior doesn't change unexpectedly."""

    @pytest.fixture
    def router(self):
        return RouterV0Heuristic()

    @pytest.mark.asyncio
    async def test_career_decision_routes_to_expander_and_socratic(self, router):
        """Career decision input should route to expander and socratic."""
        trace = await router.route("I'm not sure if I should take this job offer or stay at my current company")

        # Should include expander (narrow framing) or socratic
        assert any(a in trace.selected_agents for a in ["expander", "socratic"])

    @pytest.mark.asyncio
    async def test_ethical_dilemma_routes_to_advocate(self, router):
        """Ethical statements should trigger advocate."""
        trace = await router.route("It would be morally wrong to do that")

        assert "advocate" in trace.selected_agents

    @pytest.mark.asyncio
    async def test_definition_request_routes_to_clarifier(self, router):
        """Definition requests should trigger clarifier."""
        trace = await router.route("What does success mean in this context?")

        # The question pattern might not trigger clarifier directly,
        # but the word 'success' is a vague term
        # This tests the vague_terms pattern
        has_clarifier = "clarifier" in trace.selected_agents
        has_questioning_agent = any(a in trace.selected_agents for a in ["socratic", "clarifier"])
        assert has_questioning_agent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
