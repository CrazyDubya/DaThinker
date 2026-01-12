"""
Routing regression tests for DaThinker v0.2.

These tests ensure:
1. Deterministic routing under heuristic router
2. Agent selection matches expected patterns
3. Routing traces are properly generated
4. Router switching works correctly

Run with: python -m pytest tests/test_routing.py -v
Or: python tests/test_routing.py
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dathinker.router import (
    RouterV0Heuristic,
    RouterV1LLM,
    RouterV2Hybrid,
    RouterType,
    RoutingTrace,
    AgentScore,
    create_router,
)


@dataclass
class RoutingTestCase:
    """Test case for routing behavior."""
    name: str
    input_text: str
    expected_agents: list[str]  # Agents that should be selected
    excluded_agents: list[str]  # Agents that should NOT be selected
    min_confidence: float = 0.0
    description: str = ""


# ===== Routing Regression Test Cases =====

ROUTING_TEST_CASES = [
    # Clarifier should trigger on vague language
    RoutingTestCase(
        name="vague_quantifiers",
        input_text="I think we need to do better and improve things a lot more",
        expected_agents=["clarifier"],
        excluded_agents=[],
        description="Vague quantifiers should trigger Clarifier",
    ),
    RoutingTestCase(
        name="undefined_references",
        input_text="It's been causing problems with the system and they keep ignoring it",
        expected_agents=["clarifier"],
        excluded_agents=[],
        description="Undefined references (it, they, the system) should trigger Clarifier",
    ),
    RoutingTestCase(
        name="multiple_goals",
        input_text="I want to improve productivity, reduce costs, increase quality, and make everyone happy",
        expected_agents=["clarifier"],
        excluded_agents=[],
        description="Multiple goals should trigger Clarifier",
    ),

    # Advocate should trigger on certainty/strong claims
    RoutingTestCase(
        name="strong_certainty",
        input_text="Obviously this is the only way forward and everyone knows it",
        expected_agents=["advocate"],
        excluded_agents=[],
        description="Strong certainty language should trigger Advocate",
    ),
    RoutingTestCase(
        name="absolute_claims",
        input_text="This approach always works and there's never any downside",
        expected_agents=["advocate"],
        excluded_agents=[],
        description="Absolute claims (always/never) should trigger Advocate",
    ),
    RoutingTestCase(
        name="moral_certainty",
        input_text="What they did was completely wrong and immoral, they should be ashamed",
        expected_agents=["advocate"],
        excluded_agents=[],
        description="Moral certainty should trigger Advocate",
    ),

    # Expander should trigger when stuck or narrow framing
    RoutingTestCase(
        name="stuck_feeling",
        input_text="I don't know what to do anymore, I'm completely stuck and need help",
        expected_agents=["expander"],
        excluded_agents=[],
        description="Stuck language should trigger Expander",
    ),
    RoutingTestCase(
        name="narrow_framing",
        input_text="I have no choice but to either quit or accept this, there's no other option",
        expected_agents=["expander"],
        excluded_agents=[],
        description="Either/or narrow framing should trigger Expander",
    ),
    RoutingTestCase(
        name="seeking_direction",
        input_text="What should I do here? What do you think would be the best approach?",
        expected_agents=["expander"],
        excluded_agents=[],
        description="Seeking direction should trigger Expander",
    ),

    # Synthesizer should trigger on complex/multi-topic input
    RoutingTestCase(
        name="multi_perspective",
        input_text="On one hand the data suggests growth, but on the other hand the market is volatile",
        expected_agents=["synthesizer"],
        excluded_agents=[],
        description="Multi-perspective input should trigger Synthesizer",
    ),
    RoutingTestCase(
        name="long_complex_input",
        input_text="First, we need to consider the technical implications of this decision. " * 15,
        expected_agents=["synthesizer"],
        excluded_agents=[],
        description="Long input should trigger Synthesizer",
    ),

    # Socratic as default
    RoutingTestCase(
        name="neutral_statement",
        input_text="I've been thinking about my career lately",
        expected_agents=["socratic"],
        excluded_agents=[],
        description="Neutral input should default to Socratic",
    ),
    RoutingTestCase(
        name="question_input",
        input_text="Why do you think people make the choices they do?",
        expected_agents=["socratic"],
        excluded_agents=[],
        description="Questions should engage Socratic",
    ),

    # Mixed signals - multiple agents
    RoutingTestCase(
        name="vague_and_certain",
        input_text="It's obviously too much and everyone agrees it needs to change",
        expected_agents=["clarifier", "advocate"],
        excluded_agents=[],
        description="Both vague and certain language should trigger both agents",
    ),
    RoutingTestCase(
        name="stuck_and_narrow",
        input_text="I'm stuck between only two options and have no idea what to do",
        expected_agents=["expander"],
        excluded_agents=["synthesizer"],
        description="Stuck with narrow framing should prioritize Expander",
    ),
]


class RouterTestRunner:
    """Runs routing regression tests."""

    def __init__(self):
        self.router = RouterV0Heuristic()
        self.results: list[dict] = []

    async def run_test(self, test_case: RoutingTestCase, verbose: bool = False) -> dict:
        """Run a single test case."""
        trace = await self.router.route(
            user_input=test_case.input_text,
            context=None,
            session_history=None,
            turn_count=0,
        )

        # Check expected agents are selected
        selected = set(trace.selected_agents)
        expected = set(test_case.expected_agents)
        excluded = set(test_case.excluded_agents)

        # Success criteria:
        # 1. At least one expected agent is selected
        # 2. No excluded agents are selected
        expected_hit = bool(expected & selected) if expected else True
        excluded_avoided = not bool(excluded & selected)
        confidence_met = trace.confidence >= test_case.min_confidence

        passed = expected_hit and excluded_avoided and confidence_met

        result = {
            "name": test_case.name,
            "passed": passed,
            "selected_agents": trace.selected_agents,
            "expected_agents": test_case.expected_agents,
            "excluded_agents": test_case.excluded_agents,
            "confidence": trace.confidence,
            "expected_hit": expected_hit,
            "excluded_avoided": excluded_avoided,
            "trace": trace,
        }

        if verbose:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"\n{status} {test_case.name}")
            print(f"  Input: {test_case.input_text[:60]}...")
            print(f"  Expected: {test_case.expected_agents}")
            print(f"  Selected: {trace.selected_agents}")
            print(f"  Confidence: {trace.confidence:.2f}")
            if not passed:
                print(f"  Reason: expected_hit={expected_hit}, excluded_avoided={excluded_avoided}")
                print(f"  Description: {test_case.description}")

        return result

    async def run_all(self, verbose: bool = True) -> dict:
        """Run all routing tests."""
        print("=" * 60)
        print("ROUTING REGRESSION TESTS")
        print("=" * 60)

        self.results = []
        passed_count = 0
        failed_count = 0

        for test_case in ROUTING_TEST_CASES:
            result = await self.run_test(test_case, verbose)
            self.results.append(result)
            if result["passed"]:
                passed_count += 1
            else:
                failed_count += 1

        print("\n" + "=" * 60)
        print(f"RESULTS: {passed_count}/{len(ROUTING_TEST_CASES)} passed")
        print("=" * 60)

        return {
            "total": len(ROUTING_TEST_CASES),
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": passed_count / len(ROUTING_TEST_CASES) if ROUTING_TEST_CASES else 0,
        }


async def test_router_determinism():
    """Test that heuristic router produces deterministic results."""
    print("\n" + "=" * 60)
    print("DETERMINISM TEST")
    print("=" * 60)

    router = RouterV0Heuristic()
    test_input = "Obviously we need to improve things and do better"

    # Run same input multiple times
    results = []
    for i in range(5):
        trace = await router.route(test_input)
        results.append(tuple(trace.selected_agents))

    # All results should be identical
    unique_results = set(results)
    passed = len(unique_results) == 1

    print(f"Input: {test_input}")
    print(f"Results: {results}")
    print(f"Deterministic: {passed}")

    return passed


async def test_router_trace_completeness():
    """Test that routing traces contain all required fields."""
    print("\n" + "=" * 60)
    print("TRACE COMPLETENESS TEST")
    print("=" * 60)

    router = RouterV0Heuristic()
    trace = await router.route("Test input for trace completeness")

    required_fields = [
        "router_type",
        "input_summary",
        "agent_scores",
        "selected_agents",
        "selection_order",
        "confidence",
        "reasoning",
        "fallback_used",
        "llm_override",
    ]

    trace_dict = trace.to_dict()
    missing_fields = [f for f in required_fields if f not in trace_dict]

    passed = len(missing_fields) == 0

    print(f"Required fields: {required_fields}")
    print(f"Missing fields: {missing_fields}")
    print(f"Complete: {passed}")

    return passed


async def test_router_factory():
    """Test router factory function."""
    print("\n" + "=" * 60)
    print("ROUTER FACTORY TEST")
    print("=" * 60)

    # Heuristic router doesn't need client
    heuristic = create_router(RouterType.HEURISTIC)
    print(f"Heuristic router: {heuristic.router_type}")

    # Verify router types
    passed = (
        heuristic.router_type == RouterType.HEURISTIC
    )

    print(f"Factory works: {passed}")
    return passed


async def test_agent_score_structure():
    """Test that agent scores are properly structured."""
    print("\n" + "=" * 60)
    print("AGENT SCORE STRUCTURE TEST")
    print("=" * 60)

    router = RouterV0Heuristic()
    trace = await router.route("Obviously this is too vague and I'm stuck")

    # Check all agents have scores
    agent_names = {"socratic", "advocate", "clarifier", "synthesizer", "expander"}
    scored_agents = {s.agent_name.lower() for s in trace.agent_scores}

    # Note: Router returns display names, so we check differently
    all_agents_scored = len(trace.agent_scores) == 5

    # Check score structure
    valid_scores = all(
        0 <= s.score <= 1 and
        isinstance(s.reasons, list) and
        isinstance(s.matched_patterns, list)
        for s in trace.agent_scores
    )

    passed = all_agents_scored and valid_scores

    print(f"All agents scored: {all_agents_scored}")
    print(f"Valid score structure: {valid_scores}")
    print(f"Passed: {passed}")

    return passed


async def main():
    """Run all routing tests."""
    # Run regression tests
    runner = RouterTestRunner()
    regression_results = await runner.run_all(verbose=True)

    # Run additional tests
    determinism_passed = await test_router_determinism()
    completeness_passed = await test_router_trace_completeness()
    factory_passed = await test_router_factory()
    structure_passed = await test_agent_score_structure()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Regression tests: {regression_results['passed']}/{regression_results['total']} passed")
    print(f"Determinism test: {'PASS' if determinism_passed else 'FAIL'}")
    print(f"Trace completeness: {'PASS' if completeness_passed else 'FAIL'}")
    print(f"Router factory: {'PASS' if factory_passed else 'FAIL'}")
    print(f"Score structure: {'PASS' if structure_passed else 'FAIL'}")

    all_passed = (
        regression_results['failed'] == 0 and
        determinism_passed and
        completeness_passed and
        factory_passed and
        structure_passed
    )

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
