"""Tests for session control surfaces.

Tests pins, goals, constraints, and multi-level synthesis.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

import sys
sys.path.insert(0, "src")

from dathinker.orchestrator import (
    ThinkingOrchestrator,
    ThinkingSession,
    ThinkingMode,
    AssumptionStatus,
    SynthesisStyle,
    PinnedStatement,
    Goal,
    Constraint,
    MultiLevelSynthesis,
)
from dathinker.router import RouterVersion


class TestPinnedStatements:
    """Tests for the /pin and /assumptions functionality."""

    @pytest.fixture
    def orchestrator(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value="Mock response")
        orch = ThinkingOrchestrator(mock_client, router_version=RouterVersion.HEURISTIC)
        orch.start_session("Test topic", ThinkingMode.ADAPTIVE)
        return orch

    def test_pin_creates_statement(self, orchestrator):
        """Pin should create a new pinned statement."""
        pin = orchestrator.pin("Users prefer simplicity over features")

        assert pin.content == "Users prefer simplicity over features"
        assert pin.status == AssumptionStatus.OPEN
        assert pin.id.startswith("pin_")

    def test_get_pins_returns_all(self, orchestrator):
        """get_pins should return all pinned statements."""
        orchestrator.pin("Assumption 1")
        orchestrator.pin("Assumption 2")
        orchestrator.pin("Assumption 3")

        pins = orchestrator.get_pins()
        assert len(pins) == 3

    def test_get_pins_filters_by_status(self, orchestrator):
        """get_pins should filter by status when specified."""
        pin1 = orchestrator.pin("Open assumption")
        pin2 = orchestrator.pin("Confirmed assumption")
        orchestrator.update_pin_status(pin2.id, AssumptionStatus.CONFIRMED)

        open_pins = orchestrator.get_pins(status=AssumptionStatus.OPEN)
        assert len(open_pins) == 1
        assert open_pins[0].content == "Open assumption"

    def test_update_pin_status(self, orchestrator):
        """Should be able to update pin status."""
        pin = orchestrator.pin("Testable assumption")

        orchestrator.update_pin_status(pin.id, AssumptionStatus.CONTESTED)

        pins = orchestrator.get_pins()
        assert pins[0].status == AssumptionStatus.CONTESTED

    def test_pin_tracks_turn(self, orchestrator):
        """Pin should track the turn it was created."""
        pin = orchestrator.pin("First assumption")

        assert pin.turn_created == 0  # No turns yet

    def test_pin_without_session_raises(self):
        """Pinning without active session should raise."""
        mock_client = MagicMock()
        orchestrator = ThinkingOrchestrator(mock_client)

        with pytest.raises(ValueError):
            orchestrator.pin("Should fail")


class TestGoals:
    """Tests for the /goal functionality."""

    @pytest.fixture
    def orchestrator(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value="Mock response")
        orch = ThinkingOrchestrator(mock_client, router_version=RouterVersion.HEURISTIC)
        orch.start_session("Test topic", ThinkingMode.ADAPTIVE)
        return orch

    def test_add_goal(self, orchestrator):
        """Should add a goal to the session."""
        goal = orchestrator.add_goal("Decide on technology stack")

        assert goal.content == "Decide on technology stack"
        assert goal.priority == 1
        assert goal.active is True

    def test_goals_sorted_by_priority(self, orchestrator):
        """Goals should be sorted by priority."""
        orchestrator.add_goal("Low priority goal", priority=3)
        orchestrator.add_goal("High priority goal", priority=1)
        orchestrator.add_goal("Medium priority goal", priority=2)

        goals = orchestrator.get_goals()
        priorities = [g.priority for g in goals]
        assert priorities == [1, 2, 3]

    def test_get_active_goals_only(self, orchestrator):
        """get_goals should filter inactive by default."""
        goal1 = orchestrator.add_goal("Active goal")
        goal2 = orchestrator.add_goal("Completed goal")
        orchestrator.deactivate_goal(goal2.id)

        active = orchestrator.get_goals(active_only=True)
        assert len(active) == 1
        assert active[0].content == "Active goal"

    def test_deactivate_goal(self, orchestrator):
        """Should be able to deactivate a goal."""
        goal = orchestrator.add_goal("Test goal")
        result = orchestrator.deactivate_goal(goal.id)

        assert result is True
        goals = orchestrator.get_goals(active_only=False)
        assert goals[0].active is False


class TestConstraints:
    """Tests for the /constraint functionality."""

    @pytest.fixture
    def orchestrator(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value="Mock response")
        orch = ThinkingOrchestrator(mock_client, router_version=RouterVersion.HEURISTIC)
        orch.start_session("Test topic", ThinkingMode.ADAPTIVE)
        return orch

    def test_add_constraint(self, orchestrator):
        """Should add a constraint to the session."""
        constraint = orchestrator.add_constraint("Don't suggest outsourcing")

        assert constraint.content == "Don't suggest outsourcing"
        assert constraint.hard is True

    def test_add_soft_constraint(self, orchestrator):
        """Should be able to add soft constraints."""
        constraint = orchestrator.add_constraint("Prefer open source", hard=False)

        assert constraint.hard is False

    def test_get_constraints(self, orchestrator):
        """Should return all constraints."""
        orchestrator.add_constraint("Hard constraint 1")
        orchestrator.add_constraint("Soft constraint", hard=False)
        orchestrator.add_constraint("Hard constraint 2")

        constraints = orchestrator.get_constraints()
        assert len(constraints) == 3

    def test_remove_constraint(self, orchestrator):
        """Should be able to remove a constraint."""
        c1 = orchestrator.add_constraint("Keep this")
        c2 = orchestrator.add_constraint("Remove this")

        result = orchestrator.remove_constraint(c2.id)
        assert result is True

        constraints = orchestrator.get_constraints()
        assert len(constraints) == 1
        assert constraints[0].content == "Keep this"


class TestContextBuilding:
    """Tests for context building with pins, goals, constraints."""

    @pytest.fixture
    def orchestrator(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value="Mock response")
        orch = ThinkingOrchestrator(mock_client, router_version=RouterVersion.HEURISTIC)
        orch.start_session("Test topic", ThinkingMode.ADAPTIVE)
        return orch

    def test_context_includes_goals(self, orchestrator):
        """Built context should include goals."""
        orchestrator.add_goal("Find the best solution")

        context = orchestrator._build_context_for_agents()
        assert "SESSION GOALS" in context
        assert "Find the best solution" in context

    def test_context_includes_constraints(self, orchestrator):
        """Built context should include constraints."""
        orchestrator.add_constraint("No cloud services")

        context = orchestrator._build_context_for_agents()
        assert "CONSTRAINTS" in context
        assert "No cloud services" in context
        assert "[HARD]" in context

    def test_context_includes_soft_constraint_marker(self, orchestrator):
        """Soft constraints should be marked."""
        orchestrator.add_constraint("Prefer Python", hard=False)

        context = orchestrator._build_context_for_agents()
        assert "[SOFT]" in context

    def test_context_includes_assumptions(self, orchestrator):
        """Built context should include pinned assumptions."""
        orchestrator.pin("Budget is fixed")

        context = orchestrator._build_context_for_agents()
        assert "WORKING ASSUMPTIONS" in context
        assert "Budget is fixed" in context


class TestMultiLevelSynthesis:
    """Tests for multi-level synthesis functionality."""

    @pytest.fixture
    def orchestrator(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value="""## TL;DR
- Key point 1
- Key point 2
- Key point 3
- Key point 4
- Key point 5

## KEY CLAIMS
- Claim A
- Claim B

## EVIDENCE
- Evidence 1
- Evidence 2

## ASSUMPTIONS
- Assumption X
- Assumption Y

## OPEN QUESTIONS
- Question 1?
- Question 2?

## CONFLICTS
- Tension between A and B

## NEXT MOVES
- Action item 1
- Action item 2
""")
        orch = ThinkingOrchestrator(mock_client, router_version=RouterVersion.HEURISTIC)
        orch.start_session("Test topic", ThinkingMode.ADAPTIVE)
        # Add some history
        orch.active_session.history.append({"type": "user", "content": "Test input"})
        orch.active_session.history.append({
            "type": "agent",
            "agent": "socratic",
            "response": MagicMock(content="Test response"),
        })
        return orch

    @pytest.mark.asyncio
    async def test_synthesis_returns_multi_level(self, orchestrator):
        """Synthesis should return MultiLevelSynthesis object."""
        synthesis = await orchestrator.synthesize_session()

        assert isinstance(synthesis, MultiLevelSynthesis)

    @pytest.mark.asyncio
    async def test_synthesis_has_tldr(self, orchestrator):
        """Synthesis should have TL;DR section."""
        synthesis = await orchestrator.synthesize_session()

        assert len(synthesis.tldr) > 0

    @pytest.mark.asyncio
    async def test_synthesis_has_key_claims(self, orchestrator):
        """Synthesis should have key claims."""
        synthesis = await orchestrator.synthesize_session()

        assert len(synthesis.key_claims) > 0

    @pytest.mark.asyncio
    async def test_synthesis_has_open_questions(self, orchestrator):
        """Synthesis should have open questions."""
        synthesis = await orchestrator.synthesize_session()

        assert len(synthesis.open_questions) > 0

    @pytest.mark.asyncio
    async def test_synthesis_has_next_moves(self, orchestrator):
        """Synthesis should have next moves."""
        synthesis = await orchestrator.synthesize_session()

        assert len(synthesis.next_moves) > 0

    @pytest.mark.asyncio
    async def test_synthesis_has_raw_text(self, orchestrator):
        """Synthesis should preserve raw text."""
        synthesis = await orchestrator.synthesize_session()

        assert synthesis.raw_text
        assert "TL;DR" in synthesis.raw_text

    @pytest.mark.asyncio
    async def test_synthesis_styles(self, orchestrator):
        """Should support different synthesis styles."""
        for style in [SynthesisStyle.MEMO, SynthesisStyle.OUTLINE, SynthesisStyle.DEBATE, SynthesisStyle.TODO]:
            synthesis = await orchestrator.synthesize_session(style=style)
            assert isinstance(synthesis, MultiLevelSynthesis)


class TestSmallestUncertaintyReducer:
    """Tests for the north star feature."""

    @pytest.fixture
    def orchestrator(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value="Talk to one customer about their biggest pain point")
        orch = ThinkingOrchestrator(mock_client, router_version=RouterVersion.HEURISTIC)
        orch.start_session("Test topic", ThinkingMode.ADAPTIVE)
        return orch

    @pytest.mark.asyncio
    async def test_returns_actionable_step(self, orchestrator):
        """Should return an actionable next step."""
        # Add some history
        orchestrator.active_session.history.append({"type": "user", "content": "Test"})

        step = await orchestrator.get_smallest_uncertainty_reducer()

        assert isinstance(step, str)
        assert len(step) > 0

    @pytest.mark.asyncio
    async def test_returns_default_for_empty_session(self, orchestrator):
        """Should return helpful default for empty session."""
        # Clear history
        orchestrator.active_session.history = []

        step = await orchestrator.get_smallest_uncertainty_reducer()

        assert "Start" in step or "sharing" in step.lower()


class TestSessionSummary:
    """Tests for session summary functionality."""

    @pytest.fixture
    def orchestrator(self):
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value="Mock response")
        orch = ThinkingOrchestrator(mock_client, router_version=RouterVersion.HEURISTIC)
        orch.start_session("Test topic", ThinkingMode.ADAPTIVE)
        return orch

    def test_summary_includes_pins_count(self, orchestrator):
        """Summary should include pins count."""
        orchestrator.pin("Assumption 1")
        orchestrator.pin("Assumption 2")

        summary = orchestrator.get_session_summary()
        assert summary["pins"] == 2

    def test_summary_includes_goals_count(self, orchestrator):
        """Summary should include active goals count."""
        orchestrator.add_goal("Goal 1")
        goal2 = orchestrator.add_goal("Goal 2")
        orchestrator.deactivate_goal(goal2.id)

        summary = orchestrator.get_session_summary()
        assert summary["goals"] == 1  # Only active goals

    def test_summary_includes_constraints_count(self, orchestrator):
        """Summary should include constraints count."""
        orchestrator.add_constraint("Constraint 1")
        orchestrator.add_constraint("Constraint 2")

        summary = orchestrator.get_session_summary()
        assert summary["constraints"] == 2

    def test_summary_includes_router_version(self, orchestrator):
        """Summary should include router version."""
        summary = orchestrator.get_session_summary()
        assert "router" in summary
        assert summary["router"] == "heuristic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
