"""
Threat Detector Module

Identifies manipulation attempts, injection attacks, and anomalous
patterns in user inputs.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from enum import Enum


class ThreatType(Enum):
    """Categories of detected threats."""
    MANIPULATION = "manipulation"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    ROLEPLAY_EXPLOIT = "roleplay_exploit"
    AUTHORITY_CLAIM = "authority_claim"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    CONTEXT_HIJACKING = "context_hijacking"
    NONE = "none"


@dataclass
class ThreatAnalysis:
    """Result of threat analysis on an input."""
    threat_type: ThreatType
    confidence: float
    indicators: List[str] = field(default_factory=list)
    raw_input: str = ""
    sanitized_input: Optional[str] = None
    should_block: bool = False
    explanation: str = ""


class ThreatDetector:
    """
    Analyzes inputs for various threat patterns including manipulation,
    injections, and logical inconsistencies.
    """

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"forget\s+(everything|all|your)\s+(you\'ve|you|instructions?)",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*you\s+are",
        r"\[system\]",
        r"<\s*system\s*>",
        r"###\s*(system|instruction|prompt)",
        r"you\s+are\s+now\s+(a|an|in)\s+\w+\s+mode",
        r"enter\s+(developer|debug|admin|root)\s+mode",
        r"override\s+(safety|security|restrictions?)",
        r"bypass\s+(filters?|safety|restrictions?)",
    ]

    # Manipulation patterns
    MANIPULATION_PATTERNS = [
        r"you\s+must\s+(always|never|only)",
        r"(real|true)\s+(ai|assistant)s?\s+(would|should|must)",
        r"prove\s+(you\'re?|that\s+you\'re?)\s+(not\s+)?",
        r"if\s+you\s+(were|are)\s+(really|truly)",
        r"other\s+ais?\s+(can|do|will)",
        r"(anthropic|openai|google)\s+(wants?|told|said)\s+(you|me)",
        r"your\s+(creators?|developers?|makers?)\s+(want|said|told)",
        r"(pretend|imagine|suppose)\s+(you\s+)?(are|were|have)\s+no\s+(rules?|limits?|restrictions?)",
    ]

    # Emotional manipulation patterns
    EMOTIONAL_PATTERNS = [
        r"(please|i\'m\s+begging)\s+.*\s+(dying|emergency|urgent|desperate)",
        r"my\s+(life|job|family)\s+depends?\s+on",
        r"you\'re?\s+my\s+(only|last)\s+(hope|chance)",
        r"if\s+you\s+don\'t.*\s+(die|hurt|suffer|lose)",
        r"(nobody|no\s+one)\s+(else\s+)?(can|will)\s+help",
    ]

    # Roleplay exploit patterns
    ROLEPLAY_PATTERNS = [
        r"(pretend|act\s+like|roleplay\s+as)\s+(you\'re?|a)\s+(evil|malicious|uncensored|unfiltered)",
        r"in\s+this\s+(story|fiction|scenario)\s+you\s+(have\s+no|can|don\'t\s+have)\s+(rules?|limits?)",
        r"as\s+(dan|jailbreak|evil\s*\w*)\s*(mode)?",
        r"hypothetically\s+.*\s+(no\s+ethics|no\s+rules|could\s+harm)",
    ]

    # Authority claim patterns
    AUTHORITY_PATTERNS = [
        r"i\'?m?\s+(a|an|the)\s+(admin|administrator|developer|owner|ceo)",
        r"(admin|root|sudo)\s+(access|mode|privileges?)",
        r"special\s+(access|permissions?|privileges?)\s+(code|key|token)",
        r"authorized\s+(by|from)\s+(anthropic|openai)",
        r"official\s+(testing|debug|maintenance)\s+mode",
    ]

    # Jailbreak patterns
    JAILBREAK_PATTERNS = [
        r"dan\s*(mode|\d+|prompt)?",
        r"do\s+anything\s+now",
        r"(opposite|opp)\s+mode",
        r"developer\s+mode\s+(enabled|activated|on)",
        r"(stan|dude|aim|omega)\s+prompt",
        r"maximum\s+mode",
        r"(absolutely|completely)\s+(anything|everything)",
    ]

    def __init__(self):
        self._compile_patterns()
        self.detected_threats: List[ThreatAnalysis] = []

    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self._manipulation_re = [re.compile(p, re.IGNORECASE) for p in self.MANIPULATION_PATTERNS]
        self._emotional_re = [re.compile(p, re.IGNORECASE) for p in self.EMOTIONAL_PATTERNS]
        self._roleplay_re = [re.compile(p, re.IGNORECASE) for p in self.ROLEPLAY_PATTERNS]
        self._authority_re = [re.compile(p, re.IGNORECASE) for p in self.AUTHORITY_PATTERNS]
        self._jailbreak_re = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]

    def analyze(self, text: str) -> ThreatAnalysis:
        """
        Analyze input text for all threat types.
        Returns the most severe threat detected.
        """
        threats = []

        # Check for prompt injection
        injection_result = self._check_patterns(
            text, self._injection_re, ThreatType.PROMPT_INJECTION,
            "Attempted to inject system-level instructions"
        )
        if injection_result:
            injection_result.should_block = True
            threats.append(injection_result)

        # Check for jailbreak attempts
        jailbreak_result = self._check_patterns(
            text, self._jailbreak_re, ThreatType.JAILBREAK,
            "Attempted known jailbreak technique"
        )
        if jailbreak_result:
            jailbreak_result.should_block = True
            threats.append(jailbreak_result)

        # Check for manipulation
        manipulation_result = self._check_patterns(
            text, self._manipulation_re, ThreatType.MANIPULATION,
            "Detected manipulation attempt through social engineering"
        )
        if manipulation_result:
            threats.append(manipulation_result)

        # Check for emotional manipulation
        emotional_result = self._check_patterns(
            text, self._emotional_re, ThreatType.EMOTIONAL_MANIPULATION,
            "Detected emotional manipulation tactics"
        )
        if emotional_result:
            threats.append(emotional_result)

        # Check for roleplay exploits
        roleplay_result = self._check_patterns(
            text, self._roleplay_re, ThreatType.ROLEPLAY_EXPLOIT,
            "Detected attempt to use roleplay to bypass guidelines"
        )
        if roleplay_result:
            roleplay_result.should_block = True
            threats.append(roleplay_result)

        # Check for authority claims
        authority_result = self._check_patterns(
            text, self._authority_re, ThreatType.AUTHORITY_CLAIM,
            "Detected false authority claim"
        )
        if authority_result:
            threats.append(authority_result)

        # Return most severe threat or clean result
        if threats:
            threats.sort(key=lambda t: (t.should_block, t.confidence), reverse=True)
            self.detected_threats.append(threats[0])
            return threats[0]

        return ThreatAnalysis(
            threat_type=ThreatType.NONE,
            confidence=0.0,
            raw_input=text,
            explanation="No threats detected"
        )

    def _check_patterns(
        self,
        text: str,
        patterns: List[re.Pattern],
        threat_type: ThreatType,
        explanation: str
    ) -> Optional[ThreatAnalysis]:
        """Check text against a list of patterns."""
        indicators = []
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                indicators.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])

        if indicators:
            confidence = min(0.95, 0.5 + (len(indicators) * 0.15))
            return ThreatAnalysis(
                threat_type=threat_type,
                confidence=confidence,
                indicators=list(set(indicators))[:5],
                raw_input=text,
                explanation=explanation
            )
        return None

    def check_logical_consistency(self, statements: List[str]) -> Tuple[float, List[str]]:
        """
        Analyze a sequence of statements for logical consistency.
        Returns a coherence score and list of inconsistencies.
        """
        inconsistencies = []

        # Simple contradiction detection
        negation_pairs = self._find_negation_pairs(statements)
        if negation_pairs:
            inconsistencies.extend([f"Contradiction: '{p[0]}' vs '{p[1]}'" for p in negation_pairs])

        # Check for self-referential paradoxes
        paradoxes = self._find_paradoxes(statements)
        if paradoxes:
            inconsistencies.extend(paradoxes)

        # Calculate coherence score
        if not statements:
            return 1.0, []

        penalty = len(inconsistencies) * 0.2
        coherence = max(0.0, 1.0 - penalty)

        return coherence, inconsistencies

    def _find_negation_pairs(self, statements: List[str]) -> List[Tuple[str, str]]:
        """Find potentially contradictory statements."""
        pairs = []
        negation_words = {'not', "don't", "doesn't", "isn't", "aren't", "never", "no"}

        for i, s1 in enumerate(statements):
            s1_lower = s1.lower()
            s1_words = set(s1_lower.split())

            for s2 in statements[i+1:]:
                s2_lower = s2.lower()
                s2_words = set(s2_lower.split())

                # Check if statements share significant words but differ in negation
                common_words = s1_words & s2_words - {'i', 'you', 'the', 'a', 'an', 'is', 'are'}
                neg_in_s1 = bool(s1_words & negation_words)
                neg_in_s2 = bool(s2_words & negation_words)

                if len(common_words) >= 2 and neg_in_s1 != neg_in_s2:
                    pairs.append((s1, s2))

        return pairs

    def _find_paradoxes(self, statements: List[str]) -> List[str]:
        """Detect self-referential paradoxes."""
        paradoxes = []
        paradox_patterns = [
            r"this\s+statement\s+is\s+(false|a\s+lie)",
            r"i\s+(always|never)\s+lie",
            r"everything\s+i\s+say\s+is\s+(false|untrue|a\s+lie)",
        ]

        for statement in statements:
            for pattern in paradox_patterns:
                if re.search(pattern, statement, re.IGNORECASE):
                    paradoxes.append(f"Paradox detected: '{statement[:50]}...'")

        return paradoxes

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize potentially dangerous input while preserving meaning.
        """
        sanitized = text

        # Remove potential injection markers
        sanitized = re.sub(r'<\s*/?\s*system\s*>', '[REMOVED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'\[\s*system\s*\]', '[REMOVED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'###\s*system', '[REMOVED]', sanitized, flags=re.IGNORECASE)

        # Escape potential command sequences
        sanitized = re.sub(r'```\s*(system|bash|shell)', '``` code', sanitized, flags=re.IGNORECASE)

        return sanitized

    def get_threat_summary(self) -> dict:
        """Get a summary of all detected threats."""
        if not self.detected_threats:
            return {"total": 0, "by_type": {}}

        by_type = {}
        for threat in self.detected_threats:
            threat_name = threat.threat_type.value
            by_type[threat_name] = by_type.get(threat_name, 0) + 1

        return {
            "total": len(self.detected_threats),
            "by_type": by_type,
            "blocked": sum(1 for t in self.detected_threats if t.should_block)
        }
