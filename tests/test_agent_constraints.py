"""Tests for agent behavior constraints.

These tests protect the core philosophy: agents help think, not answer.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, "src")

from dathinker.agents import (
    AgentResponse,
    AgentIntent,
    AgentRole,
    TargetedElement,
)
from dathinker.agents.base import BaseAgent


class MockAgent(BaseAgent):
    """Mock agent for testing response parsing."""

    def __init__(self):
        self._name = "MockAgent"
        self._role = AgentRole.SOCRATIC

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> AgentRole:
        return self._role

    @property
    def system_prompt(self) -> str:
        return "You are a mock agent for testing."

    def set_role(self, role: AgentRole):
        self._role = role

    def parse_test_response(self, content: str) -> AgentResponse:
        """Expose parsing for testing."""
        return self._parse_response(content)


class TestAgentResponseStructure:
    """Tests for structured agent response."""

    @pytest.fixture
    def agent(self):
        return MockAgent()

    def test_response_has_intent(self, agent):
        """Response should have an intent field."""
        content = "Let me ask you some questions about this."
        response = agent.parse_test_response(content)

        assert hasattr(response, 'intent')
        assert isinstance(response.intent, AgentIntent)

    def test_response_has_targets(self, agent):
        """Response should have targets field."""
        content = "You're assuming that everyone agrees with this approach."
        response = agent.parse_test_response(content)

        assert hasattr(response, 'targets')
        assert isinstance(response.targets, list)

    def test_response_has_proposals(self, agent):
        """Response should have proposals field."""
        content = "**Suggestions:**\n- Consider alternative A\n- Try approach B"
        response = agent.parse_test_response(content)

        assert hasattr(response, 'proposals')
        assert isinstance(response.proposals, list)

    def test_response_to_dict(self, agent):
        """Response should be serializable to dict."""
        content = "Test content"
        response = agent.parse_test_response(content)

        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)
        assert "agent_name" in response_dict
        assert "intent" in response_dict
        assert "targets" in response_dict


class TestIntentDetermination:
    """Tests for intent determination logic."""

    @pytest.fixture
    def agent(self):
        return MockAgent()

    def test_question_intent_for_socratic(self, agent):
        """Socratic agent should default to QUESTION intent."""
        agent.set_role(AgentRole.SOCRATIC)
        content = "What do you think about this?"
        response = agent.parse_test_response(content)

        assert response.intent == AgentIntent.QUESTION

    def test_challenge_intent_for_advocate(self, agent):
        """Advocate should default to CHALLENGE intent."""
        agent.set_role(AgentRole.DEVILS_ADVOCATE)
        content = "I challenge this assumption."
        response = agent.parse_test_response(content)

        assert response.intent == AgentIntent.CHALLENGE

    def test_clarify_intent_for_clarifier(self, agent):
        """Clarifier should default to CLARIFY intent."""
        agent.set_role(AgentRole.CLARIFIER)
        content = "What do you mean by 'success'?"
        response = agent.parse_test_response(content)

        assert response.intent == AgentIntent.CLARIFY

    def test_expand_intent_for_perspective(self, agent):
        """Perspective agent should default to EXPAND intent."""
        agent.set_role(AgentRole.PERSPECTIVE)
        content = "Consider looking at this from another angle."
        response = agent.parse_test_response(content)

        assert response.intent == AgentIntent.EXPAND

    def test_synthesize_intent_for_synthesizer(self, agent):
        """Synthesizer should default to SYNTHESIZE intent."""
        agent.set_role(AgentRole.SYNTHESIZER)
        content = "Connecting these ideas together..."
        response = agent.parse_test_response(content)

        assert response.intent == AgentIntent.SYNTHESIZE


class TestTargetExtraction:
    """Tests for extracting targeted elements."""

    @pytest.fixture
    def agent(self):
        return MockAgent()

    def test_extracts_assumptions(self, agent):
        """Should extract assumptions from content."""
        content = "You're assuming that the market will grow. This assumption that growth is inevitable needs examination."
        response = agent.parse_test_response(content)

        assumption_targets = [t for t in response.targets if t.element_type == "assumption"]
        assert len(assumption_targets) > 0

    def test_extracts_claims(self, agent):
        """Should extract claims from content."""
        content = "Your claim that AI will solve everything needs scrutiny."
        response = agent.parse_test_response(content)

        claim_targets = [t for t in response.targets if t.element_type == "claim"]
        assert len(claim_targets) > 0

    def test_extracts_definitions(self, agent):
        """Should extract definition requests from content."""
        content = "What do you mean by 'success'? The term 'effective' seems unclear."
        response = agent.parse_test_response(content)

        definition_targets = [t for t in response.targets if t.element_type == "definition"]
        assert len(definition_targets) > 0

    def test_targeted_element_has_action(self, agent):
        """Targeted elements should have an action."""
        content = "You're assuming that everyone agrees."
        response = agent.parse_test_response(content)

        for target in response.targets:
            assert target.action in ["question", "challenge", "clarify", "support"]


class TestQuestionExtraction:
    """Tests for extracting questions from responses."""

    @pytest.fixture
    def agent(self):
        return MockAgent()

    def test_extracts_questions_from_bullet_list(self, agent):
        """Should extract questions from bullet point lists."""
        content = """**Questions to consider:**
- What are the main risks?
- How does this compare to alternatives?
- Who would be affected?"""
        response = agent.parse_test_response(content)

        assert len(response.questions) >= 2

    def test_extracts_numbered_questions(self, agent):
        """Should extract questions from numbered lists."""
        content = """Questions:
1. Have you considered the costs?
2. What's the timeline?
3. Who is responsible?"""
        response = agent.parse_test_response(content)

        assert len(response.questions) >= 2


class TestChallengeExtraction:
    """Tests for extracting challenges from responses."""

    @pytest.fixture
    def agent(self):
        return MockAgent()

    def test_extracts_challenges(self, agent):
        """Should extract challenges from content."""
        content = """**Challenges to your assumptions:**
- The data doesn't support this conclusion
- There are counterexamples in the literature
- The timing assumption may be wrong"""
        response = agent.parse_test_response(content)

        assert len(response.challenges) >= 2


class TestInsightExtraction:
    """Tests for extracting insights from responses."""

    @pytest.fixture
    def agent(self):
        return MockAgent()

    def test_extracts_insights(self, agent):
        """Should extract insights from content."""
        content = """**Key observations:**
- There's a pattern of risk aversion
- The stakeholders have conflicting interests
- The timeline is the binding constraint"""
        response = agent.parse_test_response(content)

        assert len(response.insights) >= 2


class TestProposalExtraction:
    """Tests for extracting proposals from responses."""

    @pytest.fixture
    def agent(self):
        return MockAgent()

    def test_extracts_proposals(self, agent):
        """Should extract proposals from content."""
        content = """**Suggestions:**
- Consider running a pilot first
- Might want to gather more data
- Could explore partnership options"""
        response = agent.parse_test_response(content)

        assert len(response.proposals) >= 2


class TestNonAnswerConstraints:
    """Tests to ensure agents don't give direct answers.

    These are behavioral constraints that protect the core philosophy.
    """

    @pytest.fixture
    def agent(self):
        return MockAgent()

    def test_socratic_must_ask_questions(self, agent):
        """Socratic agent responses should contain questions."""
        agent.set_role(AgentRole.SOCRATIC)

        # Simulate a typical Socratic response
        content = """Let me help you think through this.

**Questions to explore:**
- What assumptions are you making here?
- How would you know if this was working?
- What would change your mind?

These questions might help clarify your thinking."""

        response = agent.parse_test_response(content)

        # Socratic MUST have questions
        assert len(response.questions) > 0, "Socratic response must contain questions"
        assert response.intent == AgentIntent.QUESTION

    def test_advocate_must_challenge(self, agent):
        """Advocate responses should contain challenges."""
        agent.set_role(AgentRole.DEVILS_ADVOCATE)

        content = """Let me push back on this.

**Challenges to consider:**
- The data doesn't fully support this
- There are counterexamples
- The reasoning has gaps

What if the opposite were true?"""

        response = agent.parse_test_response(content)

        # Advocate should have challenges
        assert len(response.challenges) > 0, "Advocate response must contain challenges"

    def test_clarifier_must_clarify(self, agent):
        """Clarifier should identify vagueness."""
        agent.set_role(AgentRole.CLARIFIER)

        content = """I notice some terms that could use clarification.

**Questions about definitions:**
- What exactly do you mean by 'success'?
- How are you defining 'effective'?

Being precise about these terms would help."""

        response = agent.parse_test_response(content)

        # Clarifier should ask for definitions
        assert response.intent == AgentIntent.CLARIFY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
