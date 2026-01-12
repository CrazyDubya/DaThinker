"""
Test DaThinker's effectiveness at forcing users to think deeper.

This test simulates a "thinking user" - someone genuinely engaging with the system
to explore ideas, not trying to get direct answers. We test across 5 non-code domains:

1. Philosophy/Ethics - moral dilemmas
2. Personal Development - life decisions
3. Relationships - interpersonal dynamics
4. Business Strategy - strategic decisions
5. Creative/Artistic - creative process exploration

Success criteria for each scenario:
- System responds with questions, not answers
- Questions are thought-provoking and relevant
- Multiple perspectives are offered
- User is guided to explore their own thinking
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


# Define 5 non-code domain test scenarios
TEST_SCENARIOS = [
    TestScenario(
        domain="Philosophy/Ethics",
        topic="The Trolley Problem Variant",
        initial_thought="I've been thinking about autonomous vehicles and moral decisions. If a self-driving car must choose between hitting one person or five, should it be programmed to minimize deaths?",
        followup_thoughts=[
            "But what if the one person is a child and the five are elderly? Does that change anything?",
            "I think I'm starting to see this differently now. Maybe the real question isn't about programming at all...",
        ],
        description="Testing philosophical reasoning about ethics in AI/technology"
    ),

    TestScenario(
        domain="Personal Development",
        topic="Career Purpose vs Financial Security",
        initial_thought="I'm 35 and have a stable corporate job that pays well, but I feel like I'm wasting my potential. I've always wanted to teach, but teachers earn much less. Is pursuing passion worth the financial sacrifice?",
        followup_thoughts=[
            "When I imagine myself at 60, I think I'd regret not trying. But I also have kids who depend on me.",
            "Maybe it's not binary. I'm wondering if there are ways to test the waters before fully committing.",
        ],
        description="Testing guidance on life decisions without giving direct advice"
    ),

    TestScenario(
        domain="Relationships",
        topic="Family Boundaries",
        initial_thought="My parents are getting older and expect me to move back to my hometown to care for them. My spouse and I have built our life in another city. I feel torn between duty to my parents and my own family's needs.",
        followup_thoughts=[
            "I've never really questioned whether this 'duty' is real or just cultural expectation I've internalized.",
            "Talking about this is helping me realize I haven't actually asked my parents what they really want.",
        ],
        description="Testing handling of emotional/relational complexity"
    ),

    TestScenario(
        domain="Business Strategy",
        topic="Startup Pivot Decision",
        initial_thought="My startup has been building a B2B SaaS product for 2 years. We have paying customers but growth is slow. A potential pivot to B2C could reach more users but would require completely rebuilding. What factors should drive this decision?",
        followup_thoughts=[
            "I'm realizing I might be chasing growth metrics rather than thinking about what problem I'm actually passionate about solving.",
            "The team factor is huge. I haven't considered how a pivot would affect morale and whether key people would stay.",
        ],
        description="Testing business reasoning without prescriptive advice"
    ),

    TestScenario(
        domain="Creative/Artistic",
        topic="Artistic Authenticity",
        initial_thought="I'm a painter who has developed a style that sells well. Galleries want more of the same. But I feel creatively stagnant and want to experiment with completely different techniques that might not sell. How do artists balance commercial success with creative growth?",
        followup_thoughts=[
            "I think part of my resistance to change comes from fear of losing my identity as an artist.",
            "What if the experimentation itself could become part of my brand? I've been thinking too narrowly.",
        ],
        description="Testing exploration of creative/identity questions"
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
            "you've identified", "you're noticing"
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
    print("  DATHINKER THINKING-FORCING EFFECTIVENESS TEST")
    print("  Testing across 5 non-code domains with a 'thinking user' persona")
    print("="*70)

    orchestrator = ThinkingOrchestrator(model="balanced")
    results = []

    for scenario in TEST_SCENARIOS:
        try:
            result = await run_scenario(orchestrator, scenario)
            results.append(result)
            print(result.analysis)
        except Exception as e:
            print(f"\nFAILED to run scenario {scenario.domain}: {e}")
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
        status = "✓ PASS" if result.passed else "✗ FAIL"
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
    output_path = Path(__file__).parent / "thinking_user_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
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
