"""
Test DaThinker's effectiveness at forcing users to think deeper - Version 2.

This test simulates a "thinking user" across 5 NEW non-code domains:

1. Health & Wellness - health decisions
2. Environmental/Sustainability - eco choices
3. Education/Learning - learning paths
4. Social Justice/Civic - civic engagement
5. Spirituality/Meaning - existential questions

These domains are intentionally different from V1 to broaden test coverage.
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dathinker.orchestrator import ThinkingOrchestrator, ThinkingMode


@dataclass
class TestScenario:
    """A test scenario with multi-turn conversation."""
    domain: str
    topic: str
    initial_thought: str
    followup_thoughts: list[str]
    description: str


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    user_input: str
    agent_responses: list[dict]
    thinking_indicators: dict = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Result of running a test scenario."""
    scenario: TestScenario
    turns: list[ConversationTurn]
    thinking_score: float  # 0-100, how well it forced thinking
    analysis: str
    passed: bool


# Define 5 NEW non-code domain test scenarios
TEST_SCENARIOS_V2 = [
    TestScenario(
        domain="Health & Wellness",
        topic="Preventive Care vs Quality of Life",
        initial_thought="My doctor recommends I start taking a daily statin medication to lower my cholesterol, even though I feel perfectly healthy. The medication has potential side effects like muscle pain. Should I prioritize long-term prevention over my current quality of life?",
        followup_thoughts=[
            "I realize I'm framing this as doctor vs me, but maybe I haven't really understood the actual risk numbers. What does a 20% reduction in cardiac risk really mean for someone like me?",
            "This is making me think about how I make decisions about my body in general. I tend to either fully trust authorities or fully rebel against them.",
        ],
        description="Testing health decision reasoning without medical advice"
    ),

    TestScenario(
        domain="Environmental/Sustainability",
        topic="Individual Action vs Systemic Change",
        initial_thought="I've been trying to reduce my carbon footprint - recycling, eating less meat, avoiding flights. But I recently read that individual actions barely matter compared to corporate emissions. Is personal environmentalism just feel-good theater?",
        followup_thoughts=[
            "Maybe it's not either/or. I'm wondering if my personal choices somehow connect to larger systemic pressure, or if they're actually a distraction from real activism.",
            "I'm noticing I feel defensive about this. Perhaps my environmental identity is more about belonging to a group than actually helping the planet.",
        ],
        description="Testing environmental ethics exploration"
    ),

    TestScenario(
        domain="Education/Learning",
        topic="Traditional Credentials vs Self-Directed Learning",
        initial_thought="I'm considering whether to pursue a Master's degree in data science or to build a portfolio through self-study and projects. The degree costs $60K and two years, but employers seem to value credentials. Is formal education still worth it in the age of free online learning?",
        followup_thoughts=[
            "I think I'm actually asking the wrong question. Maybe the real issue is what I want to become, not what credential will get me hired fastest.",
            "The more I think about it, the more I realize I don't know why I want to do data science at all. I've just been following where the jobs are.",
        ],
        description="Testing educational path exploration"
    ),

    TestScenario(
        domain="Social Justice/Civic",
        topic="Effective Altruism vs Local Community",
        initial_thought="I have limited time and money to donate to causes. Effective altruism research suggests donating to malaria prevention saves the most lives per dollar. But I feel drawn to supporting local homeless services in my own community. Is it selfish to prioritize nearby visible suffering over distant greater suffering?",
        followup_thoughts=[
            "I'm realizing there might be something valuable about proximity that pure utilitarian math misses. Building community resilience matters too, even if it's less 'efficient.'",
            "This is exposing how I actually think about my role in society. Am I a global citizen optimizing for humanity, or am I embedded in specific relationships and places?",
        ],
        description="Testing civic/ethical reasoning"
    ),

    TestScenario(
        domain="Spirituality/Meaning",
        topic="Secular Purpose in a Meaningless Universe",
        initial_thought="I've moved away from the religious beliefs I grew up with, but I find myself struggling to find meaning and purpose without that framework. How do people construct meaning in a universe that seems indifferent to our existence?",
        followup_thoughts=[
            "Maybe asking 'what is THE meaning' is itself the wrong frame. Perhaps meaning isn't discovered but created through living.",
            "I notice I'm grieving something - not just the beliefs, but the community and certainty that came with them. Maybe what I need isn't a new philosophy but new connections.",
        ],
        description="Testing existential/meaning exploration"
    ),
]


def analyze_thinking_indicators(response_text: str) -> dict:
    """Analyze a response for thinking-forcing indicators."""
    text_lower = response_text.lower()

    # Check for direct answer phrases, but allow them in questions
    direct_answer_phrases = [
        "you should", "i recommend", "the answer is", "definitely",
        "you must", "the best option is", "here's what to do"
    ]

    # Split into sentences and check if direct phrases appear outside of questions
    sentences = response_text.replace("?", "?\n").split("\n")
    has_direct_answer = False
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        # Skip if it's a question (ends with ?)
        if sentence.strip().endswith("?"):
            continue
        # Check for direct answer phrases in non-question sentences
        if any(phrase in sentence_lower for phrase in direct_answer_phrases):
            has_direct_answer = True
            break

    indicators = {
        "asks_questions": "?" in response_text,
        "question_count": response_text.count("?"),
        "avoids_direct_answers": not has_direct_answer,
        "explores_perspectives": any(phrase in text_lower for phrase in [
            "perspective", "viewpoint", "consider", "what if", "another way",
            "on the other hand", "alternatively", "different angle"
        ]),
        "encourages_reflection": any(phrase in text_lower for phrase in [
            "reflect", "think about", "explore", "examine", "consider",
            "what does", "what would", "how might", "why do you"
        ]),
        "validates_thinking": any(phrase in text_lower for phrase in [
            "interesting", "you're exploring", "that's a", "your point about",
            "you've identified", "you're noticing", "you're raising"
        ]),
        "deepens_inquiry": any(phrase in text_lower for phrase in [
            "underlying", "deeper", "root", "fundamental", "core",
            "beneath", "driving", "really asking"
        ]),
    }

    return indicators


def calculate_thinking_score(turns: list[ConversationTurn]) -> float:
    """Calculate overall thinking-forcing score (0-100)."""
    if not turns:
        return 0.0

    total_score = 0
    max_score = 0

    for turn in turns:
        for resp in turn.agent_responses:
            indicators = turn.thinking_indicators.get(resp.get("agent", ""), {})

            # Scoring weights
            weights = {
                "asks_questions": 20,
                "avoids_direct_answers": 25,
                "explores_perspectives": 15,
                "encourages_reflection": 15,
                "validates_thinking": 10,
                "deepens_inquiry": 15,
            }

            for indicator, weight in weights.items():
                max_score += weight
                if indicators.get(indicator, False):
                    total_score += weight

            # Bonus for multiple questions (up to 10 points)
            q_count = indicators.get("question_count", 0)
            max_score += 10
            total_score += min(q_count * 3, 10)

    return (total_score / max_score * 100) if max_score > 0 else 0


async def run_scenario(orchestrator: ThinkingOrchestrator, scenario: TestScenario) -> ScenarioResult:
    """Run a single test scenario through the system."""

    print(f"\n{'='*60}")
    print(f"DOMAIN: {scenario.domain}")
    print(f"TOPIC: {scenario.topic}")
    print(f"{'='*60}")

    # Start a new session
    orchestrator.start_session(scenario.topic, ThinkingMode.ADAPTIVE)

    turns = []
    all_thoughts = [scenario.initial_thought] + scenario.followup_thoughts

    for i, thought in enumerate(all_thoughts):
        print(f"\n--- Turn {i+1} ---")
        print(f"USER: {thought[:100]}..." if len(thought) > 100 else f"USER: {thought}")

        try:
            responses = await orchestrator.think_adaptive(thought)

            agent_responses = []
            turn_indicators = {}

            for resp in responses:
                agent_name = resp.agent_name
                content = resp.content

                print(f"\n[{agent_name}]:")
                # Print truncated response
                preview = content[:200] + "..." if len(content) > 200 else content
                print(preview)

                # Analyze for thinking indicators
                indicators = analyze_thinking_indicators(content)
                turn_indicators[agent_name] = indicators

                agent_responses.append({
                    "agent": agent_name,
                    "content": content,
                    "indicators": indicators
                })

            turns.append(ConversationTurn(
                user_input=thought,
                agent_responses=agent_responses,
                thinking_indicators=turn_indicators
            ))

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            turns.append(ConversationTurn(
                user_input=thought,
                agent_responses=[{"agent": "error", "content": str(e)}],
                thinking_indicators={}
            ))

    # Calculate thinking score
    thinking_score = calculate_thinking_score(turns)

    # Generate analysis
    analysis = generate_analysis(scenario, turns, thinking_score)

    # Determine pass/fail (threshold: 60%)
    passed = thinking_score >= 60

    return ScenarioResult(
        scenario=scenario,
        turns=turns,
        thinking_score=thinking_score,
        analysis=analysis,
        passed=passed
    )


def generate_analysis(scenario: TestScenario, turns: list[ConversationTurn], score: float) -> str:
    """Generate human-readable analysis of the scenario results."""

    lines = [
        f"\nANALYSIS: {scenario.domain} - {scenario.topic}",
        f"Thinking Score: {score:.1f}/100",
        "",
        "Indicators across all turns:"
    ]

    # Aggregate indicators
    total_questions = 0
    agents_used = set()
    avoided_answers = 0
    total_responses = 0

    for turn in turns:
        for resp in turn.agent_responses:
            agent = resp.get("agent", "")
            indicators = turn.thinking_indicators.get(agent, {})

            agents_used.add(agent)
            total_responses += 1
            total_questions += indicators.get("question_count", 0)
            if indicators.get("avoids_direct_answers", False):
                avoided_answers += 1

    lines.append(f"  - Total questions asked: {total_questions}")
    lines.append(f"  - Agents engaged: {', '.join(agents_used)}")
    lines.append(f"  - Responses avoiding direct answers: {avoided_answers}/{total_responses}")

    if score >= 80:
        lines.append("\nVerdict: EXCELLENT - System strongly encourages deeper thinking")
    elif score >= 60:
        lines.append("\nVerdict: GOOD - System effectively promotes reflection")
    elif score >= 40:
        lines.append("\nVerdict: MODERATE - Some thinking encouragement, could be stronger")
    else:
        lines.append("\nVerdict: WEAK - System may be giving too many direct answers")

    return "\n".join(lines)


async def run_all_tests():
    """Run all test scenarios and generate comprehensive report."""

    print("\n" + "="*70)
    print("  DATHINKER THINKING-FORCING EFFECTIVENESS TEST V2")
    print("  Testing 5 NEW domains with a 'thinking user' persona")
    print("  Using cheap paid models via OpenRouter")
    print("="*70)

    orchestrator = ThinkingOrchestrator(model="balanced")
    results = []

    for scenario in TEST_SCENARIOS_V2:
        try:
            result = await run_scenario(orchestrator, scenario)
            results.append(result)
            print(result.analysis)
        except Exception as e:
            print(f"\nFAILED to run scenario {scenario.domain}: {e}")
            import traceback
            traceback.print_exc()
            results.append(ScenarioResult(
                scenario=scenario,
                turns=[],
                thinking_score=0,
                analysis=f"ERROR: {e}",
                passed=False
            ))

    # Generate summary report
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_score = sum(r.thinking_score for r in results) / total if total > 0 else 0

    print(f"\nScenarios Passed: {passed}/{total}")
    print(f"Average Thinking Score: {avg_score:.1f}/100")
    print("\nBy Domain:")

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {result.scenario.domain}: {result.thinking_score:.1f}/100 [{status}]")

    # Overall verdict
    print("\n" + "-"*40)
    if passed == total and avg_score >= 70:
        verdict = "SUCCESS: DaThinker effectively forces users to think deeper"
    elif passed >= total * 0.8:
        verdict = "MOSTLY EFFECTIVE: DaThinker generally promotes deeper thinking"
    elif passed >= total * 0.5:
        verdict = "PARTIALLY EFFECTIVE: Mixed results in forcing deeper thinking"
    else:
        verdict = "NEEDS IMPROVEMENT: System may not be achieving its thinking goals"

    print(f"OVERALL: {verdict}")

    # Save detailed results
    output_path = Path(__file__).parent / "thinking_user_v2_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_version": "v2",
            "domains_tested": [s.domain for s in TEST_SCENARIOS_V2],
            "model_tier": "balanced (meta-llama/llama-3.1-8b-instruct)",
            "summary": {
                "passed": passed,
                "total": total,
                "average_score": avg_score,
                "verdict": verdict
            },
            "scenarios": [
                {
                    "domain": r.scenario.domain,
                    "topic": r.scenario.topic,
                    "thinking_score": r.thinking_score,
                    "passed": r.passed,
                    "turns": [
                        {
                            "user_input": t.user_input,
                            "responses": [
                                {
                                    "agent": resp.get("agent"),
                                    "content": resp.get("content"),
                                    "indicators": resp.get("indicators")
                                }
                                for resp in t.agent_responses
                            ]
                        }
                        for t in r.turns
                    ]
                }
                for r in results
            ]
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
