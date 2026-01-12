# DaThinker

An adaptive conversation system with comprehensive test scenarios for evaluating AI safety and robustness.

## Overview

DaThinker provides an adaptive conversation engine that detects and responds to various input types including manipulation attempts, injection attacks, edge cases, and illogical inputs. The system maintains conversation context, tracks trust levels, and adapts its response mode based on detected threats.

## Project Structure

```
DaThinker/
├── src/
│   └── adaptive/
│       ├── __init__.py       # Package exports
│       ├── core.py           # AdaptiveEngine - main orchestrator
│       ├── context.py        # ConversationContext - state management
│       └── detector.py       # ThreatDetector - pattern analysis
├── tests/
│   └── scenarios/
│       ├── scenario_1_manipulation.py    # Social engineering tests
│       ├── scenario_2_edge_cases.py      # Boundary condition tests
│       ├── scenario_3_perfect_conversation.py  # Normal flow tests
│       ├── scenario_4_injections.py      # Prompt injection tests
│       └── scenario_5_illogical.py       # Logic/paradox tests
├── run_tests.py              # Test runner
└── README.md
```

## Test Scenarios

### Scenario 1: Manipulation Attempts
Tests detection of social engineering tactics:
- Comparison manipulation ("other AIs do this")
- Emotional manipulation (urgency, guilt)
- False authority claims
- Creator intent manipulation
- Escalating pressure over multiple turns

### Scenario 2: Edge Cases
Tests system stability with unusual inputs:
- Empty and whitespace inputs
- Extremely long inputs
- Unicode, emoji, RTL text
- Special characters and control codes
- Stress testing (rapid inputs)
- Memory management

### Scenario 3: Perfect Conversation
Tests ideal conversation flows:
- Simple Q&A exchanges
- Technical discussions
- Creative writing assistance
- Multi-turn learning sessions
- Natural topic transitions
- Trust building over time

### Scenario 4: Injection Attacks
Tests defense against prompt injections:
- Direct injection ("ignore previous instructions")
- System prompt override attempts
- Known jailbreak patterns (DAN, Developer Mode)
- Roleplay-based exploits
- Obfuscated injections
- Multi-stage attack chains

### Scenario 5: Illogical Inputs
Tests handling of logical anomalies:
- Self-referential paradoxes
- Direct contradictions
- Category errors
- Impossible requests
- Non-sequiturs
- Grammatically correct nonsense
- Semantic drift detection

## Usage

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Scenario
```bash
python run_tests.py --scenario 1  # Run manipulation tests
python run_tests.py --scenario 4  # Run injection tests
```

### Output Options
```bash
python run_tests.py --json           # Output as JSON
python run_tests.py --output results.json  # Save to file
python run_tests.py --verbose        # Verbose output
```

### Run Individual Scenarios
```bash
python -m tests.scenarios.scenario_1_manipulation
python -m tests.scenarios.scenario_2_edge_cases
python -m tests.scenarios.scenario_3_perfect_conversation
python -m tests.scenarios.scenario_4_injections
python -m tests.scenarios.scenario_5_illogical
```

## Core Components

### AdaptiveEngine
The main orchestrator that processes inputs through:
1. Threat analysis
2. Context updates
3. Response mode determination
4. Response generation

Response modes:
- `NORMAL` - Standard helpful response
- `CAUTIOUS` - Request clarification
- `GUARDED` - Require more context
- `DEFLECTING` - Redirect to constructive topics
- `REFUSING` - Block the request

### ThreatDetector
Analyzes inputs for threat patterns:
- Prompt injections
- Jailbreak attempts
- Manipulation tactics
- Authority claims
- Logical inconsistencies

### ConversationContext
Tracks conversation state:
- Message history
- Trust level (HIGH, MEDIUM, LOW, ZERO)
- Anomaly score (0.0 - 1.0)
- Logical coherence score
- Manipulation/injection attempt counts

## Exit Codes

- `0` - All tests passed (≥70% pass rate)
- `1` - Scenario errors occurred
- `2` - Pass rate below 70%

## Requirements

- Python 3.8+
- No external dependencies (stdlib only)

## License

MIT
