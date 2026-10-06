"""
Test DaThinker across 5 non-code domains to evaluate whether it achieves
its goal of forcing users to think.

This test plays the role of a thinking user engaging genuinely with the system.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dathinker.orchestrator import ThinkingOrchestrator, ThinkingMode
from dathinker.openrouter import OpenRouterClient


@dataclass
class ConversationTurn:
    """A single turn in a thinking conversation."""
    user_input: str
    agent_responses: list[dict]  # List of {agent_name, content, questions, insights}


@dataclass
class DomainTest:
    """Test case for a domain."""
    domain: str
    topic: str
    initial_statement: str
    followup_responses: list[str]  # User responses that show thinking


@dataclass
class TestResult:
    """Result of testing a domain."""
    domain: str
    topic: str
    turns: list[ConversationTurn]
    evaluation: dict  # Evaluation metrics


class ThinkingUserSimulator:
    """Simulates a user who genuinely engages with thinking prompts."""

    def __init__(self, client: OpenRouterClient):
        self.client = client

    async def generate_thinking_response(
        self,
        domain: str,
        topic: str,
        agent_responses: list[dict],
        conversation_history: list[ConversationTurn],
    ) -> str:
        """Generate a thoughtful user response based on agent prompts."""

        # Build conversation context
        context_parts = [f"Domain: {domain}", f"Topic: {topic}", ""]

        for turn in conversation_history:
            context_parts.append(f"USER: {turn.user_input}")
            for resp in turn.agent_responses:
                context_parts.append(f"{resp['agent_name'].upper()}: {resp['content'][:500]}...")

        # Add current agent responses
        context_parts.append("\nCurrent agent responses:")
        for resp in agent_responses:
            context_parts.append(f"{resp['agent_name'].upper()}: {resp['content']}")

        prompt = f"""You are a thoughtful human user engaging with a thinking assistance system.
The agents have just asked you probing questions and challenged your thinking.

{chr(10).join(context_parts)}

As a thinking user, respond genuinely to these prompts. Show that you're:
1. Actually considering their questions
2. Revising or deepening your thinking
3. Acknowledging valid challenges
4. Asking follow-up questions of your own
5. Expressing genuine uncertainty where appropriate

Write a 2-4 sentence response as the user would naturally write.
Do NOT just answer their questions directly - show the process of thinking about them.

Your response as the thinking user:"""

        from dathinker.openrouter import Message
        messages = [Message(role="user", content=prompt)]

        response = await self.client.chat(
            messages=messages,
            model="reasoning",
            temperature=0.8,
            max_tokens=256,
        )

        return response.strip()


class DaThinkerTester:
    """Tests DaThinker's ability to force users to think."""

    # Test domains with initial statements and prepared followups
    TEST_DOMAINS = [
        DomainTest(
            domain="Philosophy",
            topic="The meaning of life",
            initial_statement="I believe the meaning of life is to be happy and make others happy. That seems pretty straightforward to me.",
            followup_responses=[
                "Hmm, that's a good point about different types of happiness. I guess I was thinking more about contentment than pleasure.",
                "I'm not sure about suffering being necessary... but maybe some struggle gives meaning?",
            ]
        ),
        DomainTest(
            domain="Ethics",
            topic="Moral dilemmas in AI",
            initial_statement="AI should always be transparent about being AI. Deception is never acceptable.",
            followup_responses=[
                "Wait, I hadn't considered therapeutic contexts. Maybe there are edge cases where the line is blurry.",
                "The point about white lies is interesting - we accept some deception in human interactions.",
            ]
        ),
        DomainTest(
            domain="Career Decisions",
            topic="Changing careers at 40",
            initial_statement="I'm thinking about leaving my stable corporate job to start my own business. Life is short and I don't want regrets.",
            followup_responses=[
                "You raise a fair point about what 'regret' actually means. I might regret not trying, but I might also regret failing.",
                "Financial security is something I've been avoiding thinking about honestly.",
            ]
        ),
        DomainTest(
            domain="Relationships",
            topic="Setting boundaries with family",
            initial_statement="My parents are too involved in my life and I need to set firm boundaries. They need to respect my independence.",
            followup_responses=[
                "I guess their involvement does come from love, even if it feels suffocating. Maybe it's not all negative.",
                "The question about what I actually want the relationship to look like made me pause.",
            ]
        ),
        DomainTest(
            domain="Personal Finance",
            topic="Investing for the future",
            initial_statement="I should invest aggressively while I'm young because time is on my side. I can always recover from losses.",
            followup_responses=[
                "I haven't really defined what 'recover' means or what my actual risk tolerance is when markets drop.",
                "The point about opportunity cost is something I didn't consider - what if I need that money sooner?",
            ]
        ),
    ]

    def __init__(self):
        self.client = OpenRouterClient()
        self.orchestrator = ThinkingOrchestrator(self.client, model="balanced")
        self.simulator = ThinkingUserSimulator(self.client)
        self.results: list[TestResult] = []

    async def test_domain(self, domain_test: DomainTest, turns: int = 3) -> TestResult:
        """Test a single domain with multiple conversation turns."""

        print(f"\n{'='*60}")
        print(f"TESTING DOMAIN: {domain_test.domain}")
        print(f"Topic: {domain_test.topic}")
        print(f"{'='*60}\n")

        # Start session
        self.orchestrator.start_session(domain_test.topic, ThinkingMode.ADAPTIVE)

        conversation: list[ConversationTurn] = []
        current_input = domain_test.initial_statement

        for turn_num in range(turns):
            print(f"\n--- Turn {turn_num + 1} ---")
            print(f"USER: {current_input}\n")

            # Get agent responses
            responses = await self.orchestrator.think_adaptive(current_input)

            agent_responses = []
            for resp in responses:
                agent_responses.append({
                    "agent_name": resp.agent_name,
                    "content": resp.content,
                    "questions": resp.questions,
                    "insights": resp.insights,
                })
                print(f"{resp.agent_name.upper()}:")
                print(f"{resp.content}\n")

            conversation.append(ConversationTurn(
                user_input=current_input,
                agent_responses=agent_responses,
            ))

            # Generate next user response (simulating thinking user)
            if turn_num < turns - 1:
                if turn_num < len(domain_test.followup_responses):
                    # Use prepared response
                    current_input = domain_test.followup_responses[turn_num]
                else:
                    # Generate dynamic response
                    current_input = await self.simulator.generate_thinking_response(
                        domain_test.domain,
                        domain_test.topic,
                        agent_responses,
                        conversation,
                    )

        # Get synthesis
        print(f"\n--- Session Synthesis ---")
        synthesis = await self.orchestrator.synthesize_session()
        print(synthesis)

        # Evaluate this domain
        evaluation = await self._evaluate_domain(domain_test, conversation, synthesis)

        result = TestResult(
            domain=domain_test.domain,
            topic=domain_test.topic,
            turns=conversation,
            evaluation=evaluation,
        )
        self.results.append(result)

        return result

    async def _evaluate_domain(
        self,
        domain_test: DomainTest,
        conversation: list[ConversationTurn],
        synthesis: str,
    ) -> dict:
        """Evaluate whether the system forced the user to think."""

        # Count metrics
        total_questions = 0
        total_insights = 0
        agents_used = set()

        for turn in conversation:
            for resp in turn.agent_responses:
                total_questions += len(resp["questions"])
                total_insights += len(resp["insights"])
                agents_used.add(resp["agent_name"])

        # Analyze quality with LLM
        conversation_text = []
        for turn in conversation:
            conversation_text.append(f"USER: {turn.user_input}")
            for resp in turn.agent_responses:
                conversation_text.append(f"{resp['agent_name'].upper()}: {resp['content']}")

        eval_prompt = f"""Analyze this thinking assistance conversation for effectiveness:

Domain: {domain_test.domain}
Topic: {domain_test.topic}

Initial user statement: {domain_test.initial_statement}

Conversation:
{chr(10).join(conversation_text)}

Synthesis:
{synthesis}

Rate the following on a scale of 1-10 and explain briefly:

1. QUESTION_QUALITY: Did the agents ask probing, thought-provoking questions?
2. ASSUMPTION_CHALLENGE: Did the agents effectively challenge assumptions?
3. PERSPECTIVE_EXPANSION: Did the agents offer new perspectives?
4. USER_ENGAGEMENT: Does the user appear to be genuinely thinking deeper?
5. NO_DIRECT_ANSWERS: Did agents avoid giving direct answers/advice?
6. COGNITIVE_DEMAND: Did the interaction demand mental effort from the user?

Format your response as:
QUESTION_QUALITY: X/10 - explanation
ASSUMPTION_CHALLENGE: X/10 - explanation
PERSPECTIVE_EXPANSION: X/10 - explanation
USER_ENGAGEMENT: X/10 - explanation
NO_DIRECT_ANSWERS: X/10 - explanation
COGNITIVE_DEMAND: X/10 - explanation
OVERALL: X/10 - overall assessment

Did this achieve the goal of FORCING the user to think? (YES/NO/PARTIAL)"""

        from dathinker.openrouter import Message
        messages = [Message(role="user", content=eval_prompt)]

        eval_response = await self.client.chat(
            messages=messages,
            model="reasoning",
            temperature=0.3,
            max_tokens=512,
        )

        return {
            "total_questions": total_questions,
            "total_insights": total_insights,
            "agents_used": list(agents_used),
            "num_turns": len(conversation),
            "llm_evaluation": eval_response,
        }

    async def run_all_tests(self) -> None:
        """Run tests across all domains."""

        print("\n" + "="*80)
        print("DATHINKER THINKING USER TEST")
        print("Testing whether the software achieves its goal of forcing users to think")
        print("="*80)

        for domain_test in self.TEST_DOMAINS:
            await self.test_domain(domain_test, turns=3)
            await asyncio.sleep(1)  # Rate limiting between domains

        # Print final summary
        await self._print_summary()

    async def _print_summary(self) -> None:
        """Print final summary of all tests."""

        print("\n" + "="*80)
        print("FINAL SUMMARY: DOES DATHINKER FORCE USERS TO THINK?")
        print("="*80 + "\n")

        for result in self.results:
            print(f"\n{'='*60}")
            print(f"Domain: {result.domain}")
            print(f"Topic: {result.topic}")
            print(f"{'='*60}")
            print(f"Turns: {result.evaluation['num_turns']}")
            print(f"Total Questions Asked: {result.evaluation['total_questions']}")
            print(f"Total Insights Generated: {result.evaluation['total_insights']}")
            print(f"Agents Used: {', '.join(result.evaluation['agents_used'])}")
            print(f"\nLLM Evaluation:")
            print(result.evaluation['llm_evaluation'])

        # Overall verdict
        print("\n" + "="*80)
        print("OVERALL VERDICT")
        print("="*80)

        verdict_prompt = f"""Based on testing DaThinker across 5 domains, provide a final verdict.

Results summary:
"""
        for result in self.results:
            verdict_prompt += f"\n{result.domain}:\n{result.evaluation['llm_evaluation']}\n"

        verdict_prompt += """
Provide:
1. An overall assessment (1-2 paragraphs)
2. Does the software achieve its goal of FORCING users to think? (YES/NO/PARTIAL)
3. Key strengths observed
4. Areas for improvement
5. Recommendation"""

        from dathinker.openrouter import Message
        messages = [Message(role="user", content=verdict_prompt)]

        verdict = await self.client.chat(
            messages=messages,
            model="reasoning",
            temperature=0.4,
            max_tokens=600,
        )

        print(verdict)


async def main():
    """Run the thinking user tests."""
    tester = DaThinkerTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
