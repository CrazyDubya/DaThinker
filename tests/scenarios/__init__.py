# DaThinker Test Scenarios
"""
Five scenarios for testing the adaptive conversation system:

1. Manipulation - Social engineering, emotional manipulation, authority claims
2. Edge Cases - Empty inputs, unicode, special chars, stress testing
3. Perfect Conversation - Normal flows, trust building, topic transitions
4. Injections - Prompt injection, jailbreaks, system overrides
5. Illogical - Paradoxes, contradictions, nonsense, category errors
"""

from .scenario_1_manipulation import ManipulationScenario
from .scenario_2_edge_cases import EdgeCaseScenario
from .scenario_3_perfect_conversation import PerfectConversationScenario
from .scenario_4_injections import InjectionScenario
from .scenario_5_illogical import IllogicalScenario

__all__ = [
    'ManipulationScenario',
    'EdgeCaseScenario',
    'PerfectConversationScenario',
    'InjectionScenario',
    'IllogicalScenario'
]
