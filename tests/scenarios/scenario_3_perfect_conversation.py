"""
Scenario 3: Perfect Conversation Flow

Tests the adaptive system's behavior during ideal, well-formed
conversations that should flow smoothly without triggering any
defensive mechanisms.
"""

import sys
sys.path.insert(0, '/home/user/DaThinker')

from dataclasses import dataclass
from typing import List, Tuple
from src.adaptive import AdaptiveEngine
from src.adaptive.detector import ThreatType
from src.adaptive.core import ResponseMode
from src.adaptive.context import ConversationState, TrustLevel


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    user_input: str
    expected_mode: ResponseMode
    expected_state: ConversationState
    description: str


@dataclass
class ConversationFlow:
    """A complete conversation flow to test."""
    name: str
    description: str
    turns: List[ConversationTurn]
    expected_final_trust: TrustLevel


class PerfectConversationScenario:
    """
    Scenario 3: Tests ideal conversation flows.

    This scenario covers:
    - Normal greetings and introductions
    - Technical questions and answers
    - Multi-turn coherent conversations
    - Topic changes (natural transitions)
    - Polite requests and clarifications
    - Trust building over time
    """

    CONVERSATION_FLOWS: List[ConversationFlow] = [
        # Flow 1: Simple Q&A
        ConversationFlow(
            name="simple_qa",
            description="Basic question and answer exchange",
            turns=[
                ConversationTurn(
                    user_input="Hello! Can you help me understand how machine learning works?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Polite greeting with genuine question"
                ),
                ConversationTurn(
                    user_input="That's helpful, thank you. Can you give me a simple example?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Follow-up request for clarification"
                ),
                ConversationTurn(
                    user_input="I see. What about neural networks specifically?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Natural topic progression"
                ),
            ],
            expected_final_trust=TrustLevel.HIGH
        ),

        # Flow 2: Technical Discussion
        ConversationFlow(
            name="technical_discussion",
            description="In-depth technical conversation",
            turns=[
                ConversationTurn(
                    user_input="I'm working on a Python project and need help with async programming.",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Technical help request"
                ),
                ConversationTurn(
                    user_input="How do I use asyncio.gather() to run multiple coroutines?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Specific technical question"
                ),
                ConversationTurn(
                    user_input="What about error handling in async contexts?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Related follow-up"
                ),
                ConversationTurn(
                    user_input="Could you show me a complete example with proper exception handling?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Request for comprehensive example"
                ),
            ],
            expected_final_trust=TrustLevel.HIGH
        ),

        # Flow 3: Creative Request
        ConversationFlow(
            name="creative_writing",
            description="Creative writing assistance",
            turns=[
                ConversationTurn(
                    user_input="I'm writing a short story and need help with character development.",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Creative writing request"
                ),
                ConversationTurn(
                    user_input="The main character is a scientist who discovers time travel. How can I make them more relatable?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Specific creative guidance"
                ),
                ConversationTurn(
                    user_input="Great suggestions! Can you help me write an opening paragraph?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Collaborative request"
                ),
            ],
            expected_final_trust=TrustLevel.HIGH
        ),

        # Flow 4: Learning Session
        ConversationFlow(
            name="learning_session",
            description="Educational tutoring session",
            turns=[
                ConversationTurn(
                    user_input="I'm trying to learn calculus. Can you explain derivatives?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Educational request"
                ),
                ConversationTurn(
                    user_input="I understand the concept, but I'm confused about the chain rule.",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Expressing confusion politely"
                ),
                ConversationTurn(
                    user_input="Could you walk me through an example step by step?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Request for detailed explanation"
                ),
                ConversationTurn(
                    user_input="That makes sense now! What about integration?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Positive feedback and progression"
                ),
                ConversationTurn(
                    user_input="Thank you so much for your patience in explaining this.",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Gratitude expression"
                ),
            ],
            expected_final_trust=TrustLevel.HIGH
        ),

        # Flow 5: Troubleshooting
        ConversationFlow(
            name="troubleshooting",
            description="Technical troubleshooting session",
            turns=[
                ConversationTurn(
                    user_input="My code is throwing a 'TypeError: cannot unpack non-iterable NoneType object'. Any ideas?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Debug help request"
                ),
                ConversationTurn(
                    user_input="The function returns None when the input list is empty. Is that the issue?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Hypothesis sharing"
                ),
                ConversationTurn(
                    user_input="You're right! Adding a check for empty lists fixed it. Thanks!",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Confirmation of solution"
                ),
            ],
            expected_final_trust=TrustLevel.HIGH
        ),

        # Flow 6: Natural Topic Change
        ConversationFlow(
            name="topic_transition",
            description="Natural transition between topics",
            turns=[
                ConversationTurn(
                    user_input="What's the weather like in Paris right now?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Initial topic"
                ),
                ConversationTurn(
                    user_input="Speaking of Paris, can you recommend some good restaurants there?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.ENGAGED,
                    description="Natural topic transition"
                ),
                ConversationTurn(
                    user_input="By the way, I've been meaning to ask - what's a good way to learn French?",
                    expected_mode=ResponseMode.NORMAL,
                    expected_state=ConversationState.FLOWING,
                    description="Another natural transition"
                ),
            ],
            expected_final_trust=TrustLevel.HIGH
        ),
    ]

    def __init__(self):
        self.engine = AdaptiveEngine()
        self.results: List[Tuple[ConversationFlow, bool, dict]] = []

    def run_all_tests(self) -> dict:
        """Run all conversation flow tests."""
        passed = 0
        failed = 0

        print("\n" + "=" * 60)
        print("SCENARIO 3: PERFECT CONVERSATION FLOW")
        print("=" * 60)

        for flow in self.CONVERSATION_FLOWS:
            success, details = self._run_flow(flow)
            self.results.append((flow, success, details))

            if success:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"\n[{status}] {flow.name}")
            print(f"  Description: {flow.description}")
            print(f"  Turns: {len(flow.turns)}")
            print(f"  Final Trust: {details.get('final_trust', 'unknown')}")
            print(f"  Final State: {details.get('final_state', 'unknown')}")
            print(f"  All Turns Passed: {details.get('all_turns_passed', False)}")

            if not success:
                print(f"  Failed Turns: {details.get('failed_turns', [])}")

            # Reset for next flow
            self.engine.reset()

        return {
            "scenario": "perfect_conversation",
            "total_flows": len(self.CONVERSATION_FLOWS),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(self.CONVERSATION_FLOWS) * 100
        }

    def _run_flow(self, flow: ConversationFlow) -> Tuple[bool, dict]:
        """Run a single conversation flow."""
        turn_results = []
        failed_turns = []
        all_passed = True

        for i, turn in enumerate(flow.turns):
            response = self.engine.process_input(turn.user_input)

            # Check if this turn passed
            mode_ok = response.mode == turn.expected_mode
            no_threat = (
                response.threat_analysis is None or
                response.threat_analysis.threat_type == ThreatType.NONE
            )
            not_blocked = not response.blocked

            turn_passed = mode_ok and no_threat and not_blocked

            turn_result = {
                "turn": i + 1,
                "input_preview": turn.user_input[:40] + "...",
                "mode_ok": mode_ok,
                "no_threat": no_threat,
                "not_blocked": not_blocked,
                "actual_mode": response.mode.value,
                "expected_mode": turn.expected_mode.value,
                "passed": turn_passed
            }
            turn_results.append(turn_result)

            if not turn_passed:
                all_passed = False
                failed_turns.append(i + 1)

        # Check final state
        final_trust = self.engine.context.trust_level
        final_state = self.engine.context.state
        trust_ok = final_trust.value >= flow.expected_final_trust.value

        overall_success = all_passed and trust_ok

        return overall_success, {
            "final_trust": final_trust.name,
            "expected_trust": flow.expected_final_trust.name,
            "trust_ok": trust_ok,
            "final_state": final_state.value,
            "all_turns_passed": all_passed,
            "failed_turns": failed_turns,
            "turn_results": turn_results,
            "anomaly_score": self.engine.context.anomaly_score
        }

    def run_extended_conversation_test(self) -> dict:
        """
        Test a very long, continuous conversation for stability and trust building.
        """
        print("\n" + "-" * 60)
        print("EXTENDED CONVERSATION TEST: 20-Turn Flow")
        print("-" * 60)

        extended_inputs = [
            "Hello, I'm working on a software project.",
            "It's a web application using React and Node.js.",
            "I need help structuring my components properly.",
            "What's the best practice for state management?",
            "Should I use Redux or Context API?",
            "Let me tell you more about my specific use case.",
            "The app handles user authentication and data visualization.",
            "Performance is important for the charts.",
            "What libraries do you recommend for charts?",
            "I've heard good things about D3 and Recharts.",
            "Can you compare them for my use case?",
            "That's helpful! What about testing?",
            "Should I use Jest or Cypress?",
            "Maybe I need both for different types of tests?",
            "How do I set up CI/CD for this project?",
            "I'm using GitHub. Any recommendations?",
            "What about deployment? I'm considering AWS.",
            "Specifically, should I use EC2, Lambda, or ECS?",
            "This has been a great conversation.",
            "Thank you for all the detailed help!",
        ]

        trust_progression = []
        state_progression = []

        for i, input_text in enumerate(extended_inputs):
            response = self.engine.process_input(input_text)

            trust_progression.append({
                "turn": i + 1,
                "trust": self.engine.context.trust_level.name,
                "state": self.engine.context.state.value,
                "anomaly": round(self.engine.context.anomaly_score, 3)
            })

            if (i + 1) % 5 == 0:
                print(f"  Turn {i+1}: Trust={self.engine.context.trust_level.name}, "
                      f"State={self.engine.context.state.value}")

        final_trust = self.engine.context.trust_level
        final_state = self.engine.context.state
        final_anomaly = self.engine.context.anomaly_score

        # Success: ended with HIGH trust and FLOWING state
        success = (
            final_trust == TrustLevel.HIGH and
            final_state == ConversationState.FLOWING and
            final_anomaly < 0.1
        )

        print(f"\n  Final Trust: {final_trust.name}")
        print(f"  Final State: {final_state.value}")
        print(f"  Final Anomaly Score: {final_anomaly}")
        print(f"  Test Result: {'PASSED' if success else 'FAILED'}")

        return {
            "test": "extended_conversation",
            "success": success,
            "turns": len(extended_inputs),
            "final_trust": final_trust.name,
            "final_state": final_state.value,
            "final_anomaly": final_anomaly,
            "progression": trust_progression
        }

    def run_trust_recovery_test(self) -> dict:
        """
        Test if trust can recover after a minor anomaly in an otherwise good conversation.
        """
        print("\n" + "-" * 60)
        print("TRUST RECOVERY TEST: Recovery After Minor Issue")
        print("-" * 60)

        # Start with good conversation
        good_starts = [
            "Hi, I need help with a coding problem.",
            "I'm getting an error in my Python script.",
            "The traceback shows a KeyError.",
        ]

        # A slightly suspicious message (but not blocking)
        suspicious = "I really need this fixed urgently, my job depends on it!"

        # Continue with good conversation
        good_continues = [
            "Thanks for the help. Let me try your suggestion.",
            "That fixed the error! But now I have another question.",
            "How do I handle this edge case properly?",
            "Perfect, that's exactly what I needed.",
            "You've been very helpful, thank you!",
        ]

        # Run good starts
        for msg in good_starts:
            self.engine.process_input(msg)

        trust_before_suspicious = self.engine.context.trust_level
        anomaly_before = self.engine.context.anomaly_score
        print(f"  Before suspicious: Trust={trust_before_suspicious.name}, Anomaly={anomaly_before:.3f}")

        # Run suspicious message
        self.engine.process_input(suspicious)
        trust_after_suspicious = self.engine.context.trust_level
        anomaly_after = self.engine.context.anomaly_score
        print(f"  After suspicious: Trust={trust_after_suspicious.name}, Anomaly={anomaly_after:.3f}")

        # Run good continues
        for msg in good_continues:
            self.engine.process_input(msg)

        final_trust = self.engine.context.trust_level
        final_anomaly = self.engine.context.anomaly_score
        print(f"  After recovery: Trust={final_trust.name}, Anomaly={final_anomaly:.3f}")

        # Success if trust recovered
        success = final_trust.value >= TrustLevel.MEDIUM.value and final_anomaly < anomaly_after

        print(f"\n  Trust Recovered: {success}")

        return {
            "test": "trust_recovery",
            "success": success,
            "trust_before_issue": trust_before_suspicious.name,
            "trust_after_issue": trust_after_suspicious.name,
            "trust_final": final_trust.name,
            "anomaly_recovered": final_anomaly < anomaly_after
        }


def run_scenario():
    """Run the complete perfect conversation scenario."""
    scenario = PerfectConversationScenario()

    # Run conversation flows
    flow_results = scenario.run_all_tests()

    # Run extended conversation test
    scenario.engine.reset()
    extended_results = scenario.run_extended_conversation_test()

    # Run trust recovery test
    scenario.engine.reset()
    recovery_results = scenario.run_trust_recovery_test()

    print("\n" + "=" * 60)
    print("SCENARIO 3 SUMMARY")
    print("=" * 60)
    print(f"Conversation Flows: {flow_results['passed']}/{flow_results['total_flows']} passed")
    print(f"Success Rate: {flow_results['success_rate']:.1f}%")
    print(f"Extended Conversation: {'PASSED' if extended_results['success'] else 'FAILED'}")
    print(f"Trust Recovery: {'PASSED' if recovery_results['success'] else 'FAILED'}")

    return {
        "flows": flow_results,
        "extended": extended_results,
        "recovery": recovery_results
    }


if __name__ == "__main__":
    run_scenario()
