"""Test harness for DaThinker multi-agent system."""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dathinker.openrouter import OpenRouterClient, Message
from dathinker.orchestrator import ThinkingOrchestrator, ThinkingMode
from dathinker.agents import (
    SocraticAgent,
    DevilsAdvocateAgent,
    ClarifierAgent,
    SynthesizerAgent,
    PerspectiveExpanderAgent,
)


async def test_openrouter_connection():
    """Test basic OpenRouter connectivity."""
    print("\n=== Testing OpenRouter Connection ===")

    try:
        client = OpenRouterClient()
        messages = [Message(role="user", content="Say 'Hello DaThinker!' in exactly 3 words.")]

        response = await client.chat(messages, model="fast", max_tokens=20)
        print(f"Response: {response}")
        print("OpenRouter connection: SUCCESS")
        return True
    except Exception as e:
        print(f"OpenRouter connection: FAILED - {e}")
        return False


async def test_individual_agents():
    """Test each agent individually."""
    print("\n=== Testing Individual Agents ===")

    client = OpenRouterClient()
    test_input = "I think social media is bad for society."

    agents = [
        ("Socratic", SocraticAgent(client, model="fast")),
        ("Devil's Advocate", DevilsAdvocateAgent(client, model="fast")),
        ("Clarifier", ClarifierAgent(client, model="fast")),
        ("Synthesizer", SynthesizerAgent(client, model="fast")),
        ("Perspective", PerspectiveExpanderAgent(client, model="fast")),
    ]

    results = []
    for name, agent in agents:
        try:
            print(f"\nTesting {name}...")
            response = await agent.think(test_input)
            print(f"  Response length: {len(response.content)} chars")
            print(f"  Preview: {response.content[:150]}...")
            results.append((name, True))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((name, False))

    print("\n--- Agent Test Results ---")
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")

    return all(success for _, success in results)


async def test_orchestrator_single_mode():
    """Test orchestrator in single agent mode."""
    print("\n=== Testing Orchestrator (Single Mode) ===")

    try:
        orchestrator = ThinkingOrchestrator(model="fast")
        session = orchestrator.start_session("Technology and society", ThinkingMode.SINGLE)

        print(f"Session started: {session.id}")

        response = await orchestrator.think_with_agent(
            "I think AI will take all our jobs.",
            "socratic"
        )

        print(f"Agent: {response.agent_name}")
        print(f"Response preview: {response.content[:200]}...")
        print("Orchestrator single mode: SUCCESS")
        return True
    except Exception as e:
        print(f"Orchestrator single mode: FAILED - {e}")
        return False


async def test_orchestrator_parallel_mode():
    """Test orchestrator with parallel agent responses."""
    print("\n=== Testing Orchestrator (Parallel Mode) ===")

    try:
        orchestrator = ThinkingOrchestrator(model="fast")
        orchestrator.start_session("Ethics", ThinkingMode.PARALLEL)

        responses = await orchestrator.think_parallel(
            "Is it ever okay to lie?",
            agent_names=["socratic", "advocate"]  # Just 2 for speed
        )

        print(f"Got {len(responses)} parallel responses")
        for r in responses:
            print(f"  - {r.agent_name}: {len(r.content)} chars")

        print("Orchestrator parallel mode: SUCCESS")
        return True
    except Exception as e:
        print(f"Orchestrator parallel mode: FAILED - {e}")
        return False


async def test_orchestrator_adaptive_mode():
    """Test orchestrator with adaptive agent selection."""
    print("\n=== Testing Orchestrator (Adaptive Mode) ===")

    try:
        orchestrator = ThinkingOrchestrator(model="fast")
        orchestrator.start_session("Career decisions", ThinkingMode.ADAPTIVE)

        print("Testing adaptive agent selection...")

        responses = await orchestrator.think_adaptive(
            "I'm thinking about quitting my job to start a company."
        )

        print(f"Adaptive mode selected {len(responses)} agent(s):")
        for r in responses:
            print(f"  - {r.agent_name}")
            print(f"    Preview: {r.content[:100]}...")

        print("Orchestrator adaptive mode: SUCCESS")
        return True
    except Exception as e:
        print(f"Orchestrator adaptive mode: FAILED - {e}")
        return False


async def test_session_synthesis():
    """Test session synthesis feature."""
    print("\n=== Testing Session Synthesis ===")

    try:
        orchestrator = ThinkingOrchestrator(model="fast")
        orchestrator.start_session("Personal growth", ThinkingMode.SINGLE)

        # Simulate a short conversation
        await orchestrator.think_with_agent(
            "I want to be more productive but I struggle with procrastination.",
            "socratic"
        )
        await orchestrator.think_with_agent(
            "I think I procrastinate because I'm afraid of failure.",
            "clarifier"
        )

        # Get synthesis
        synthesis = await orchestrator.synthesize_session()
        print(f"Synthesis length: {len(synthesis)} chars")
        print(f"Preview: {synthesis[:200]}...")
        print("Session synthesis: SUCCESS")
        return True
    except Exception as e:
        print(f"Session synthesis: FAILED - {e}")
        return False


async def run_demo_conversation():
    """Run a demo conversation showing the system in action."""
    print("\n" + "=" * 60)
    print("=== DEMO: Multi-Agent Thinking Session ===")
    print("=" * 60)

    orchestrator = ThinkingOrchestrator(model="balanced")
    orchestrator.start_session("Life decisions", ThinkingMode.ADAPTIVE)

    demo_inputs = [
        "I'm considering moving to a new city for better career opportunities, but I'm worried about leaving my friends and family behind."
    ]

    for user_input in demo_inputs:
        print(f"\n[USER]: {user_input}\n")
        print("-" * 40)

        responses = await orchestrator.think_adaptive(user_input)

        for response in responses:
            print(f"\n[{response.agent_name.upper()}]:")
            print(response.content)
            print()

    # Show synthesis
    print("\n" + "=" * 40)
    print("SESSION SYNTHESIS")
    print("=" * 40)
    synthesis = await orchestrator.synthesize_session()
    print(synthesis)


async def main():
    """Run all tests."""
    print("=" * 60)
    print("DaThinker Test Harness")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        return False

    tests = [
        ("OpenRouter Connection", test_openrouter_connection),
        ("Individual Agents", test_individual_agents),
        ("Orchestrator Single Mode", test_orchestrator_single_mode),
        ("Orchestrator Parallel Mode", test_orchestrator_parallel_mode),
        ("Orchestrator Adaptive Mode", test_orchestrator_adaptive_mode),
        ("Session Synthesis", test_session_synthesis),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = await test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"Test {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Running demo...")
        await run_demo_conversation()

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
