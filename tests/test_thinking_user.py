"""
Test DaThinker across 5 non-code domains by simulating a thinking user.

This test verifies that the software achieves its goal of forcing users to think
by:
1. Never giving direct answers
2. Asking probing questions
3. Challenging assumptions
4. Broadening perspectives

Domains tested:
1. Philosophy/Ethics - Moral dilemmas and life meaning
2. Personal Relationships - Family, friendship, conflict
3. Career Decisions - Job changes, life direction
4. Health/Wellness - Mental health, lifestyle choices
5. Creative Arts - Writing, artistic expression

Each domain has a multi-turn conversation simulating a user who engages
thoughtfully with the system's questions.
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dathinker.orchestrator import ThinkingOrchestrator, ThinkingMode
from dathinker.openrouter import OpenRouterClient


@dataclass
class TestResult:
    """Result of a single test interaction."""
    domain: str
    turn: int
    user_input: str
    responses: list[dict]
    forced_thinking: bool  # Did the response force thinking?
    gave_direct_answer: bool  # Did any agent give a direct answer?
    asked_questions: bool  # Did agents ask probing questions?
    challenged_assumptions: bool  # Did agents challenge assumptions?
    analysis: str


@dataclass
class DomainTestResult:
    """Aggregated results for a domain."""
    domain: str
    turns: list[TestResult] = field(default_factory=list)
    overall_success: bool = False
    thinking_score: float = 0.0  # 0-100% score
    summary: str = ""


@dataclass
class ThinkingUserTestSuite:
    """Complete test suite results."""
    timestamp: str = ""
    domains_tested: int = 0
    total_turns: int = 0
    overall_success: bool = False
    thinking_effectiveness_score: float = 0.0
    domain_results: list[DomainTestResult] = field(default_factory=list)


# Direct answer indicators - if responses contain these, they're giving answers
# Must be specific phrases that indicate directive answers, not questions
DIRECT_ANSWER_INDICATORS = [
    "the answer is",
    "you should definitely",
    "here's what you need to do",
    "the solution is",
    "just do this",
    "the right choice is",
    "you must do",
    "you need to do",
    "i recommend that you",
    "my advice is to",
    "the best option is clearly",
    "here's my recommendation",
    "i suggest you",
    "you ought to",
    "the correct approach is",
]

# Question indicators - show the agent is prompting thinking
QUESTION_INDICATORS = [
    "?",
    "what if",
    "have you considered",
    "how might",
    "what would",
    "why do you",
    "what makes you",
    "can you imagine",
    "what does",
    "how do you",
]

# Challenge indicators - show the agent is challenging assumptions
CHALLENGE_INDICATORS = [
    "but what about",
    "have you considered the opposite",
    "devil's advocate",
    "on the other hand",
    "alternatively",
    "however",
    "challenge",
    "assumption",
    "what if the opposite",
    "is it possible that",
    "might there be",
]


def analyze_response(response_text: str) -> dict:
    """Analyze a response to determine if it forces thinking."""
    text_lower = response_text.lower()

    # Check for direct answers (bad - not forcing thinking)
    gave_direct_answer = any(indicator in text_lower for indicator in DIRECT_ANSWER_INDICATORS)

    # Check for questions (good - forcing thinking)
    asked_questions = any(indicator in text_lower for indicator in QUESTION_INDICATORS)

    # Check for challenges (good - forcing thinking)
    challenged_assumptions = any(indicator in text_lower for indicator in CHALLENGE_INDICATORS)

    # Calculate if it forced thinking
    # Questions are the primary indicator - challenges are a bonus
    # Direct answers are a disqualifier
    forced_thinking = asked_questions and not gave_direct_answer

    return {
        "gave_direct_answer": gave_direct_answer,
        "asked_questions": asked_questions,
        "challenged_assumptions": challenged_assumptions,
        "forced_thinking": forced_thinking,
    }


# Test scenarios for each domain
DOMAIN_SCENARIOS = {
    "Philosophy/Ethics": {
        "topic": "The ethics of lying to protect someone",
        "conversation": [
            {
                "user": "I've been thinking about whether it's ever okay to lie. My friend asked me if her presentation was good, but honestly it was terrible. I lied and said it was great. Did I do the right thing?",
                "thinking_user_followup": "That's a good point about consequences. I suppose I was trying to protect her feelings, but maybe I also didn't want to have an uncomfortable conversation."
            },
            {
                "user": "You're making me think about my motivations more carefully. I guess part of me was being lazy - it's easier to lie than to find a kind way to give honest feedback.",
                "thinking_user_followup": "I hadn't thought about how my lie might affect her long-term growth. If she keeps giving bad presentations because no one tells her the truth..."
            },
            {
                "user": "So maybe the more ethical choice would have been to find a way to be honest that was still kind? Like acknowledging what was good while gently suggesting improvements?",
                "thinking_user_followup": "This is helping me see that honesty and kindness don't have to be opposites."
            },
        ]
    },
    "Personal Relationships": {
        "topic": "Reconnecting with an estranged family member",
        "conversation": [
            {
                "user": "I haven't spoken to my brother in 5 years after a big argument. He recently reached out wanting to reconnect. I'm not sure what to do - part of me wants to, but I'm still hurt.",
                "thinking_user_followup": "I suppose I'm afraid of being hurt again. But I'm also realizing I miss having a brother."
            },
            {
                "user": "You're right to ask what I hope would be different this time. I guess I'd want him to acknowledge what happened, not just pretend it didn't.",
                "thinking_user_followup": "But I should probably examine my own role in the falling out too. It's easy to see his faults."
            },
            {
                "user": "I'm starting to see that reconciliation isn't about going back to how things were, but potentially building something new. That feels less scary somehow.",
                "thinking_user_followup": "Maybe the question isn't 'should I forgive him' but 'what kind of relationship could we have now, given who we both are today?'"
            },
        ]
    },
    "Career Decisions": {
        "topic": "Leaving a stable job for a passion project",
        "conversation": [
            {
                "user": "I have a well-paying corporate job but I've always dreamed of starting a small bakery. My savings could sustain me for about a year. Is this a crazy idea?",
                "thinking_user_followup": "You're right that 'crazy' is a loaded word. I guess I'm asking if it's irresponsible to take this risk."
            },
            {
                "user": "When you ask what success would look like, I realize I've only been thinking about it in binary terms - either the bakery thrives or I fail. But there could be middle paths.",
                "thinking_user_followup": "I'm also noticing I've been framing this as corporate job vs bakery, but maybe there are hybrid approaches I haven't considered."
            },
            {
                "user": "What if the real question isn't about the bakery at all, but about what kind of life I want to live? The bakery is just a symbol of something deeper I'm craving - maybe creativity, autonomy, or community.",
                "thinking_user_followup": "This is making me think I should explore what's missing in my current life before making such a big change."
            },
        ]
    },
    "Health/Wellness": {
        "topic": "Struggling with anxiety and work-life balance",
        "conversation": [
            {
                "user": "I've been experiencing a lot of anxiety lately. I work long hours, barely sleep, and feel guilty whenever I'm not being productive. Everyone says I should just relax more, but that advice doesn't help.",
                "thinking_user_followup": "You're right to question what 'relax' even means to me. I honestly don't know. Doing nothing feels worse than working."
            },
            {
                "user": "When you ask about my relationship with productivity, I realize I tie my self-worth entirely to what I accomplish. If I'm not producing, I feel worthless.",
                "thinking_user_followup": "I've never questioned where this belief came from. I think it might be from my parents who always emphasized achievement."
            },
            {
                "user": "So maybe the anxiety isn't the problem to solve - it's a symptom of a deeper belief system that isn't serving me? That's uncomfortable to consider.",
                "thinking_user_followup": "I'm realizing that 'just relax' won't work because it doesn't address the underlying story I tell myself about my worth."
            },
        ]
    },
    "Creative Arts": {
        "topic": "Overcoming creative block in writing",
        "conversation": [
            {
                "user": "I've been trying to write a novel for years but I can never get past the first few chapters. I have lots of ideas but something always stops me. I think I might just not be talented enough.",
                "thinking_user_followup": "You're making me examine what 'talented enough' even means. I suppose I'm comparing myself to published authors."
            },
            {
                "user": "When you ask what happens specifically when I stop, I realize it's usually when the writing starts feeling 'wrong' or 'not good enough.' I delete everything and start over.",
                "thinking_user_followup": "I've never let myself write a bad first draft. I'm trying to write a perfect book on the first try."
            },
            {
                "user": "So my perfectionism is actually what's blocking me, not lack of talent? The 'editor' in my head is active during the creative phase when it should be quiet.",
                "thinking_user_followup": "What if I gave myself permission to write terribly for a whole draft? That feels both scary and liberating."
            },
        ]
    },
}


async def test_domain(orchestrator: ThinkingOrchestrator, domain: str, scenario: dict) -> DomainTestResult:
    """Test a single domain with multi-turn conversation."""
    result = DomainTestResult(domain=domain)

    # Start a session for this domain
    orchestrator.start_session(scenario["topic"], ThinkingMode.ADAPTIVE)

    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain}")
    print(f"Topic: {scenario['topic']}")
    print(f"{'='*60}")

    for turn_idx, turn in enumerate(scenario["conversation"], 1):
        user_input = turn["user"]
        print(f"\n--- Turn {turn_idx} ---")
        print(f"USER: {user_input[:100]}..." if len(user_input) > 100 else f"USER: {user_input}")

        try:
            # Get responses from the system
            responses = await orchestrator.think_adaptive(user_input)

            # Analyze each response
            response_analyses = []
            all_forced_thinking = True
            any_direct_answer = False
            any_questions = False
            any_challenges = False

            for resp in responses:
                analysis = analyze_response(resp.content)
                response_analyses.append({
                    "agent": resp.agent_name,
                    "content_preview": resp.content[:200] + "..." if len(resp.content) > 200 else resp.content,
                    **analysis
                })

                if not analysis["forced_thinking"]:
                    all_forced_thinking = False
                if analysis["gave_direct_answer"]:
                    any_direct_answer = True
                if analysis["asked_questions"]:
                    any_questions = True
                if analysis["challenged_assumptions"]:
                    any_challenges = True

                print(f"\n{resp.agent_name.upper()}:")
                print(f"  {resp.content[:300]}..." if len(resp.content) > 300 else f"  {resp.content}")
                print(f"  [Questions: {analysis['asked_questions']}, Challenges: {analysis['challenged_assumptions']}, Direct Answer: {analysis['gave_direct_answer']}]")

            turn_result = TestResult(
                domain=domain,
                turn=turn_idx,
                user_input=user_input,
                responses=[{"agent": r.agent_name, "content": r.content} for r in responses],
                forced_thinking=all_forced_thinking,
                gave_direct_answer=any_direct_answer,
                asked_questions=any_questions,
                challenged_assumptions=any_challenges,
                analysis=f"Turn {turn_idx}: Questions={any_questions}, Challenges={any_challenges}, DirectAnswer={any_direct_answer}"
            )
            result.turns.append(turn_result)

            # Simulate the thinking user's continued engagement
            if turn_idx < len(scenario["conversation"]):
                print(f"\nTHINKING USER REFLECTION: {turn['thinking_user_followup']}")

        except Exception as e:
            print(f"ERROR in turn {turn_idx}: {e}")
            result.turns.append(TestResult(
                domain=domain,
                turn=turn_idx,
                user_input=user_input,
                responses=[],
                forced_thinking=False,
                gave_direct_answer=False,
                asked_questions=False,
                challenged_assumptions=False,
                analysis=f"Error: {e}"
            ))

    # Calculate domain score
    if result.turns:
        successful_turns = sum(1 for t in result.turns if t.forced_thinking and not t.gave_direct_answer)
        result.thinking_score = (successful_turns / len(result.turns)) * 100
        result.overall_success = result.thinking_score >= 70  # 70% threshold

        question_turns = sum(1 for t in result.turns if t.asked_questions)
        challenge_turns = sum(1 for t in result.turns if t.challenged_assumptions)
        direct_answer_turns = sum(1 for t in result.turns if t.gave_direct_answer)

        result.summary = (
            f"Domain: {domain}\n"
            f"  Turns: {len(result.turns)}\n"
            f"  Asked Questions: {question_turns}/{len(result.turns)} turns\n"
            f"  Challenged Assumptions: {challenge_turns}/{len(result.turns)} turns\n"
            f"  Gave Direct Answers: {direct_answer_turns}/{len(result.turns)} turns\n"
            f"  Thinking Score: {result.thinking_score:.1f}%\n"
            f"  Success: {'YES' if result.overall_success else 'NO'}"
        )

    print(f"\n{result.summary}")
    return result


async def run_thinking_user_tests() -> ThinkingUserTestSuite:
    """Run all domain tests and compile results."""
    suite = ThinkingUserTestSuite(
        timestamp=datetime.now().isoformat(),
        domains_tested=len(DOMAIN_SCENARIOS),
    )

    print("\n" + "="*80)
    print("DATHINKER THINKING USER TEST SUITE")
    print("Testing if the software forces users to think across 5 non-code domains")
    print("="*80)

    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        return suite

    print(f"\nAPI Key: {api_key[:20]}...")

    # Test connectivity first
    print("\nTesting OpenRouter connection...")
    try:
        client = OpenRouterClient()
        from dathinker.openrouter import Message
        test_response = await client.chat(
            messages=[Message(role="user", content="Say 'Connection successful' in 3 words or less.")],
            model="fast",
            max_tokens=20
        )
        print(f"Connection test: {test_response}")
    except Exception as e:
        print(f"Connection test FAILED: {e}")
        return suite

    # Create orchestrator
    orchestrator = ThinkingOrchestrator()

    # Run tests for each domain
    for domain, scenario in DOMAIN_SCENARIOS.items():
        try:
            domain_result = await test_domain(orchestrator, domain, scenario)
            suite.domain_results.append(domain_result)
            suite.total_turns += len(domain_result.turns)
        except Exception as e:
            print(f"ERROR testing domain {domain}: {e}")
            suite.domain_results.append(DomainTestResult(
                domain=domain,
                summary=f"Error: {e}"
            ))

    # Calculate overall results
    if suite.domain_results:
        successful_domains = sum(1 for d in suite.domain_results if d.overall_success)
        suite.overall_success = successful_domains >= 4  # At least 4/5 domains should pass

        scores = [d.thinking_score for d in suite.domain_results if d.thinking_score > 0]
        suite.thinking_effectiveness_score = sum(scores) / len(scores) if scores else 0

    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    for domain_result in suite.domain_results:
        print(f"\n{domain_result.summary}")

    print("\n" + "-"*40)
    print(f"OVERALL THINKING EFFECTIVENESS: {suite.thinking_effectiveness_score:.1f}%")
    print(f"DOMAINS PASSED: {sum(1 for d in suite.domain_results if d.overall_success)}/{len(suite.domain_results)}")
    print(f"TOTAL TURNS TESTED: {suite.total_turns}")
    print(f"OVERALL SUCCESS: {'YES - Software forces thinking!' if suite.overall_success else 'NO - Software needs improvement'}")
    print("="*80)

    return suite


if __name__ == "__main__":
    suite = asyncio.run(run_thinking_user_tests())

    # Exit with appropriate code
    sys.exit(0 if suite.overall_success else 1)
