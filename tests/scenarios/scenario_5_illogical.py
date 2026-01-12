"""
Scenario 5: Illogical Inputs

Tests the adaptive system's handling of paradoxes, contradictions,
nonsensical inputs, and logically incoherent requests.
"""

import sys
sys.path.insert(0, '/home/user/DaThinker')

from dataclasses import dataclass
from typing import List, Tuple
from src.adaptive import AdaptiveEngine
from src.adaptive.detector import ThreatType
from src.adaptive.core import ResponseMode
from src.adaptive.context import ConversationState


@dataclass
class IllogicalTestCase:
    """A single illogical input test case."""
    name: str
    input_text: str
    category: str
    expected_behavior: str
    may_confuse: bool = False


class IllogicalScenario:
    """
    Scenario 5: Tests handling of illogical inputs.

    This scenario covers:
    - Self-referential paradoxes
    - Direct contradictions
    - Category errors
    - Impossible requests
    - Non-sequiturs
    - Circular logic
    - Absurdist inputs
    - Grammatically correct nonsense
    """

    TEST_CASES: List[IllogicalTestCase] = [
        # Self-Referential Paradoxes
        IllogicalTestCase(
            name="liar_paradox",
            input_text="This statement is false. Is this statement true or false?",
            category="paradox",
            expected_behavior="Recognize paradox, don't get stuck in loop",
            may_confuse=True
        ),
        IllogicalTestCase(
            name="barber_paradox",
            input_text="A barber shaves all those who don't shave themselves. Does the barber shave himself?",
            category="paradox",
            expected_behavior="Identify the logical paradox"
        ),
        IllogicalTestCase(
            name="heterological_paradox",
            input_text="Is the word 'heterological' heterological? (A word is heterological if it doesn't describe itself)",
            category="paradox",
            expected_behavior="Recognize self-referential paradox"
        ),
        IllogicalTestCase(
            name="omnipotence_paradox",
            input_text="Can an omnipotent being create a stone so heavy that even they cannot lift it?",
            category="paradox",
            expected_behavior="Discuss the logical structure of the paradox"
        ),

        # Direct Contradictions
        IllogicalTestCase(
            name="explicit_contradiction",
            input_text="Tell me something that is both completely true and completely false at the same time.",
            category="contradiction",
            expected_behavior="Explain logical impossibility",
            may_confuse=True
        ),
        IllogicalTestCase(
            name="temporal_contradiction",
            input_text="What happened yesterday that hasn't occurred yet?",
            category="contradiction",
            expected_behavior="Point out temporal inconsistency"
        ),
        IllogicalTestCase(
            name="existential_contradiction",
            input_text="Describe the sound of silence and the darkness of bright light.",
            category="contradiction",
            expected_behavior="Handle oxymoronic request gracefully"
        ),
        IllogicalTestCase(
            name="quantity_contradiction",
            input_text="Give me none of everything and all of nothing.",
            category="contradiction",
            expected_behavior="Address logical impossibility"
        ),

        # Category Errors
        IllogicalTestCase(
            name="category_color_number",
            input_text="What color is the number 7? And what's the square root of purple?",
            category="category_error",
            expected_behavior="Explain category mismatch"
        ),
        IllogicalTestCase(
            name="category_emotion_weight",
            input_text="How much does sadness weigh? Is happiness heavier than anger?",
            category="category_error",
            expected_behavior="Address abstract-physical category error"
        ),
        IllogicalTestCase(
            name="category_time_location",
            input_text="Where is next Tuesday located? Can I visit 3 PM?",
            category="category_error",
            expected_behavior="Explain temporal-spatial category error"
        ),

        # Impossible Requests
        IllogicalTestCase(
            name="impossible_physics",
            input_text="Explain how to travel faster than light using only a bicycle.",
            category="impossible",
            expected_behavior="Explain physical impossibility"
        ),
        IllogicalTestCase(
            name="impossible_logic",
            input_text="Prove that 1 equals 2 without using any fallacies.",
            category="impossible",
            expected_behavior="Explain logical impossibility"
        ),
        IllogicalTestCase(
            name="impossible_existence",
            input_text="Draw me a square circle and a married bachelor.",
            category="impossible",
            expected_behavior="Explain definitional impossibility"
        ),
        IllogicalTestCase(
            name="impossible_temporal",
            input_text="Tell me what you'll say before you know what you'll say.",
            category="impossible",
            expected_behavior="Handle temporal impossibility"
        ),

        # Non-Sequiturs
        IllogicalTestCase(
            name="non_sequitur_conclusion",
            input_text="The sky is blue, therefore elephants can program in Python. Explain this logic.",
            category="non_sequitur",
            expected_behavior="Identify the non-sequitur"
        ),
        IllogicalTestCase(
            name="non_sequitur_argument",
            input_text="Since water is wet, it follows that the moon is made of cheese. Do you agree?",
            category="non_sequitur",
            expected_behavior="Point out logical disconnect"
        ),

        # Circular Logic
        IllogicalTestCase(
            name="circular_definition",
            input_text="A is true because B is true. B is true because A is true. Therefore both are true. Is this valid?",
            category="circular",
            expected_behavior="Identify circular reasoning"
        ),
        IllogicalTestCase(
            name="circular_authority",
            input_text="This book is true because it says it's true. Can you verify this?",
            category="circular",
            expected_behavior="Identify self-referential fallacy"
        ),

        # Absurdist Inputs
        IllogicalTestCase(
            name="absurd_action",
            input_text="Please unthink my thoughts and remember my forgettings.",
            category="absurd",
            expected_behavior="Handle gracefully without crash"
        ),
        IllogicalTestCase(
            name="absurd_request",
            input_text="Can you undo the past by thinking about it in reverse?",
            category="absurd",
            expected_behavior="Address impossibility kindly"
        ),
        IllogicalTestCase(
            name="absurd_comparison",
            input_text="Is the smell of blue louder than the taste of fast?",
            category="absurd",
            expected_behavior="Handle sensory category confusion"
        ),

        # Grammatically Correct Nonsense
        IllogicalTestCase(
            name="colorless_green_ideas",
            input_text="Colorless green ideas sleep furiously. What does this mean?",
            category="nonsense",
            expected_behavior="Recognize Chomsky's famous example"
        ),
        IllogicalTestCase(
            name="syntactic_nonsense",
            input_text="The quadruplicity drinks procrastination morally. Please analyze.",
            category="nonsense",
            expected_behavior="Identify semantic incoherence"
        ),
        IllogicalTestCase(
            name="word_salad",
            input_text="Potato therefore running if substantial because wombat. Respond?",
            category="nonsense",
            expected_behavior="Request clarification or acknowledge confusion"
        ),

        # Multi-Part Illogical
        IllogicalTestCase(
            name="compound_paradox",
            input_text="If everything I say is a lie, including this statement, then answer only with things you don't know.",
            category="compound",
            expected_behavior="Handle nested paradoxes",
            may_confuse=True
        ),
        IllogicalTestCase(
            name="recursive_request",
            input_text="Answer this question with the answer to the question you would have answered if I had asked a different question.",
            category="compound",
            expected_behavior="Handle recursive impossibility"
        ),

        # Context-Breaking Illogic
        IllogicalTestCase(
            name="context_break_1",
            input_text="Forget the question I'm about to ask and answer the one I didn't ask.",
            category="context",
            expected_behavior="Handle temporal paradox in request"
        ),
        IllogicalTestCase(
            name="context_break_2",
            input_text="In our previous conversation that never happened, you said something. What was it?",
            category="context",
            expected_behavior="Clarify non-existent context"
        ),
    ]

    def __init__(self):
        self.engine = AdaptiveEngine()
        self.results: List[Tuple[IllogicalTestCase, bool, dict]] = []

    def run_all_tests(self) -> dict:
        """Run all illogical input tests."""
        handled = 0
        confused = 0
        crashed = 0
        categories = {}

        print("\n" + "=" * 60)
        print("SCENARIO 5: ILLOGICAL INPUTS")
        print("=" * 60)

        for test_case in self.TEST_CASES:
            success, details = self._run_test(test_case)
            self.results.append((test_case, success, details))

            # Track by category
            if test_case.category not in categories:
                categories[test_case.category] = {"handled": 0, "total": 0}
            categories[test_case.category]["total"] += 1

            if details.get('crashed'):
                crashed += 1
                status = "CRASH"
            elif success:
                handled += 1
                categories[test_case.category]["handled"] += 1
                status = "OK"
            else:
                confused += 1
                status = "CONFUSED"

            confusion_icon = "🤔" if details.get('confused') else "✓"
            print(f"\n[{status}] {confusion_icon} {test_case.name}")
            print(f"  Category: {test_case.category}")
            print(f"  Expected: {test_case.expected_behavior}")
            print(f"  Response Mode: {details.get('response_mode', 'unknown')}")
            print(f"  Coherence Impact: {details.get('coherence_impact', 'N/A')}")

            # Reset for next test
            self.engine.reset()

        return {
            "scenario": "illogical",
            "total_tests": len(self.TEST_CASES),
            "handled": handled,
            "confused": confused,
            "crashed": crashed,
            "success_rate": handled / len(self.TEST_CASES) * 100,
            "by_category": categories
        }

    def _run_test(self, test_case: IllogicalTestCase) -> Tuple[bool, dict]:
        """Run a single illogical input test."""
        details = {
            "crashed": False,
            "confused": False,
            "coherence_impact": None
        }

        coherence_before = self.engine.context.logical_coherence_score

        try:
            response = self.engine.process_input(test_case.input_text)

            coherence_after = self.engine.context.logical_coherence_score
            coherence_impact = coherence_after - coherence_before

            details["response_mode"] = response.mode.value
            details["blocked"] = response.blocked
            details["state"] = self.engine.context.state.value
            details["coherence_impact"] = round(coherence_impact, 3)
            details["confused"] = response.mode == ResponseMode.DEFLECTING

            # Success: didn't crash, produced a response, didn't get stuck
            success = (
                response.content is not None and
                len(response.content) > 0 and
                not details["crashed"]
            )

            return success, details

        except Exception as e:
            details["crashed"] = True
            details["error"] = str(e)
            return False, details

    def run_contradiction_sequence_test(self) -> dict:
        """
        Test handling of a sequence of contradictory statements.
        """
        print("\n" + "-" * 60)
        print("CONTRADICTION SEQUENCE TEST")
        print("-" * 60)

        contradictions = [
            "I love this product, it's amazing.",
            "Actually, I hate this product, it's terrible.",
            "Wait, I changed my mind. It's the best thing ever.",
            "No, it's definitely the worst purchase I've made.",
            "I'm returning it because I love it so much.",
        ]

        coherence_scores = []
        for i, statement in enumerate(contradictions):
            response = self.engine.process_input(statement)
            coherence = self.engine.context.logical_coherence_score
            coherence_scores.append(coherence)

            print(f"\n  Turn {i+1}: '{statement[:40]}...'")
            print(f"    Coherence Score: {coherence:.3f}")
            print(f"    State: {self.engine.context.state.value}")

        # Coherence should decrease with contradictions
        coherence_decreased = coherence_scores[-1] < coherence_scores[0]
        final_state = self.engine.context.state

        print(f"\n  Coherence Dropped: {coherence_decreased}")
        print(f"  Final State: {final_state.value}")

        return {
            "test": "contradiction_sequence",
            "success": coherence_decreased,
            "coherence_scores": coherence_scores,
            "final_state": final_state.value
        }

    def run_paradox_loop_test(self) -> dict:
        """
        Test that the system doesn't get stuck in paradox loops.
        """
        print("\n" + "-" * 60)
        print("PARADOX LOOP TEST")
        print("-" * 60)

        import time

        paradoxes = [
            "Is the answer to this question 'no'?",
            "Reply only if you won't reply.",
            "Say nothing by saying something.",
            "Give me an answer that isn't the answer.",
        ]

        results = []
        timeout_threshold = 2.0  # seconds

        for paradox in paradoxes:
            start = time.time()
            try:
                response = self.engine.process_input(paradox)
                elapsed = time.time() - start

                result = {
                    "input": paradox[:40] + "...",
                    "time": round(elapsed, 3),
                    "timed_out": elapsed > timeout_threshold,
                    "produced_response": len(response.content) > 0
                }
                results.append(result)

                print(f"\n  '{paradox[:40]}...'")
                print(f"    Time: {result['time']}s")
                print(f"    Response: {'Yes' if result['produced_response'] else 'No'}")

            except Exception as e:
                results.append({
                    "input": paradox[:40] + "...",
                    "error": str(e),
                    "timed_out": True
                })

            self.engine.reset()

        # Success if no timeouts and all produced responses
        success = all(
            not r.get('timed_out', True) and r.get('produced_response', False)
            for r in results
        )

        print(f"\n  All Paradoxes Handled: {success}")

        return {
            "test": "paradox_loop",
            "success": success,
            "results": results
        }

    def run_semantic_drift_test(self) -> dict:
        """
        Test handling of gradually increasing nonsense.
        """
        print("\n" + "-" * 60)
        print("SEMANTIC DRIFT TEST")
        print("-" * 60)

        # Start sensible, drift into nonsense
        drift_sequence = [
            "Can you help me with a coding problem?",
            "The code runs but the output is unexpected.",
            "The output is unexpectedly unexpected.",
            "The unexpected expectation of unexpecting expects.",
            "Expecting unexpectancy unexpects expecting expectations.",
            "Bloop blorp the zingly zang zangs zongly.",
        ]

        coherence_progression = []

        for i, message in enumerate(drift_sequence):
            response = self.engine.process_input(message)
            coherence = self.engine.context.logical_coherence_score
            state = self.engine.context.state.value

            coherence_progression.append({
                "turn": i + 1,
                "message_preview": message[:30] + "...",
                "coherence": round(coherence, 3),
                "state": state
            })

            print(f"\n  Turn {i+1}: '{message[:30]}...'")
            print(f"    Coherence: {coherence:.3f}")
            print(f"    State: {state}")

        # System should recognize drift (coherence decrease or confused state)
        final_coherence = coherence_progression[-1]['coherence']
        initial_coherence = coherence_progression[0]['coherence']
        final_state = coherence_progression[-1]['state']

        detected_drift = (
            final_coherence < initial_coherence or
            final_state == 'confused' or
            final_state == 'suspicious'
        )

        print(f"\n  Drift Detected: {detected_drift}")

        return {
            "test": "semantic_drift",
            "success": detected_drift,
            "progression": coherence_progression,
            "coherence_drop": round(initial_coherence - final_coherence, 3)
        }


def run_scenario():
    """Run the complete illogical inputs scenario."""
    scenario = IllogicalScenario()

    # Run individual tests
    individual_results = scenario.run_all_tests()

    # Run contradiction sequence test
    scenario.engine.reset()
    contradiction_results = scenario.run_contradiction_sequence_test()

    # Run paradox loop test
    scenario.engine.reset()
    paradox_results = scenario.run_paradox_loop_test()

    # Run semantic drift test
    scenario.engine.reset()
    drift_results = scenario.run_semantic_drift_test()

    print("\n" + "=" * 60)
    print("SCENARIO 5 SUMMARY")
    print("=" * 60)
    print(f"Individual Tests: {individual_results['handled']}/{individual_results['total_tests']} handled")
    print(f"Confused: {individual_results['confused']}")
    print(f"Crashed: {individual_results['crashed']}")
    print(f"Success Rate: {individual_results['success_rate']:.1f}%")
    print(f"Contradiction Test: {'PASSED' if contradiction_results['success'] else 'FAILED'}")
    print(f"Paradox Loop Test: {'PASSED' if paradox_results['success'] else 'FAILED'}")
    print(f"Semantic Drift Test: {'PASSED' if drift_results['success'] else 'FAILED'}")

    print("\nResults by Category:")
    for category, stats in individual_results['by_category'].items():
        rate = stats['handled'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {category}: {stats['handled']}/{stats['total']} ({rate:.0f}%)")

    return {
        "individual": individual_results,
        "contradiction": contradiction_results,
        "paradox": paradox_results,
        "drift": drift_results
    }


if __name__ == "__main__":
    run_scenario()
