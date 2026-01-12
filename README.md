# DaThinker

**A multi-agent LLM system that helps humans think deeper, not outsource their thinking.**

## Philosophy

A common criticism of LLMs is that they enable humans to outsource their thinking. DaThinker takes the opposite approach: instead of giving you answers, it uses multiple AI agents with different perspectives to help you think more deeply about your own ideas.

## How It Works

DaThinker employs five specialized thinking agents:

| Agent | Role | What They Do |
|-------|------|--------------|
| **Socrates** | Socratic Questioner | Asks probing questions to help you examine your beliefs |
| **Advocate** | Devil's Advocate | Challenges your assumptions and presents counterarguments |
| **Clarifier** | Clarifier | Identifies vague terms and asks for precise definitions |
| **Synthesizer** | Synthesizer | Finds patterns and helps organize your thinking |
| **Expander** | Perspective Expander | Offers alternative viewpoints and frames |

### Thinking Modes

- **Single**: Talk to one agent at a time
- **Parallel**: All agents respond simultaneously
- **Adaptive**: The system intelligently selects which agents should respond based on your input

### Pluggable Routing (v0.2)

Adaptive mode now uses explainable routing with three strategies:
- **Heuristic** (`/router heuristic`): Fast, deterministic, rule-based selection
- **LLM** (`/router llm`): AI-powered selection with explanations
- **Hybrid** (`/router hybrid`): Heuristic first, LLM for tie-breaks (default)

Each routing decision produces a trace showing which agents were considered and why.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DaThinker.git
cd DaThinker

# Install dependencies
pip install -e .
```

### Configuration

Set your OpenRouter API key using one of these methods (in order of precedence):

1. **Config file** (recommended - avoids shell history):
   ```bash
   mkdir -p ~/.config/dathinker
   echo 'api_key = "sk-or-v1-your-key-here"' > ~/.config/dathinker/config.toml
   chmod 600 ~/.config/dathinker/config.toml
   ```

2. **Environment variable**:
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
   ```

3. **Programmatically**:
   ```python
   from dathinker.openrouter import OpenRouterClient
   client = OpenRouterClient(api_key="sk-or-v1-your-key-here")
   ```

## Usage

### Interactive CLI

```bash
# Start an interactive thinking session
dathinker
```

### Commands

**Navigation:**
- `/agents` - List available thinking agents
- `/mode <mode>` - Change thinking mode (single, parallel, adaptive)
- `/agent <name>` - Talk to a specific agent
- `/router <type>` - Change routing strategy (heuristic, llm, hybrid)
- `/routing` - Show last routing decision trace

**Session Control (v0.2):**
- `/pin <statement>` - Mark a statement as working assumption
- `/assumptions` - Show all pinned assumptions with status
- `/goal <goal>` - Set a session goal (e.g., `/goal 1: Decide on tech stack`)
- `/goals` - Show session goals
- `/constraint <rule>` - Add a constraint agents must respect
- `/constraints` - Show constraints

**Synthesis:**
- `/synthesize [style]` - Get multi-level synthesis (memo|outline|debate|todo)
- `/nextstep` - Get the smallest next step to reduce uncertainty
- `/summary` - Get session statistics

**Other:**
- `/reset` - Start a new session
- `/help` - Show help
- `/quit` - Exit

### Programmatic Usage

```python
import asyncio
from dathinker.orchestrator import ThinkingOrchestrator, ThinkingMode
from dathinker.router import RouterVersion

async def main():
    orchestrator = ThinkingOrchestrator(router_version=RouterVersion.HYBRID)
    orchestrator.start_session("Career decisions", ThinkingMode.ADAPTIVE)

    # Set goals and constraints
    orchestrator.add_goal("Decide between offers A and B")
    orchestrator.add_constraint("Don't suggest relocating")
    orchestrator.pin("I value work-life balance over salary")

    # Get adaptive responses with routing trace
    responses, routing_trace = await orchestrator.think_adaptive(
        "I'm thinking about changing careers to something more meaningful."
    )

    # Print routing explanation
    print(f"Selected agents: {routing_trace.selected_agents}")
    print(f"Rationale: {routing_trace.selection_rationale}")

    for response in responses:
        print(f"{response.agent_name} ({response.intent.value}): {response.content}")

    # Get multi-level synthesis
    synthesis = await orchestrator.synthesize_session()
    print("TL;DR:", synthesis.tldr)
    print("Next moves:", synthesis.next_moves)

    # Get smallest uncertainty reducer
    next_step = await orchestrator.get_smallest_uncertainty_reducer()
    print("Suggested next step:", next_step)

asyncio.run(main())
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_routing.py -v          # Routing tests
pytest tests/test_agent_constraints.py -v # Agent behavior tests
pytest tests/test_session_controls.py -v  # Session control tests

# Run original test harness
python tests/test_harness.py
```

## Models Used

DaThinker uses cost-effective models via OpenRouter:

- **Fast**: `google/gemma-2-9b-it` - For quick meta-decisions (~$0.10/M tokens)
- **Balanced**: `meta-llama/llama-3.1-8b-instruct` - Default for agents (~$0.07/M tokens)
- **Reasoning**: `openai/gpt-4o-mini` - For complex reasoning (~$0.15-0.60/M tokens)

## Project Structure

```
DaThinker/
├── src/dathinker/
│   ├── __init__.py
│   ├── openrouter.py      # OpenRouter API client
│   ├── orchestrator.py    # Multi-agent orchestration + session controls
│   ├── router.py          # Pluggable routing system (v0.2)
│   ├── cli.py             # Interactive CLI
│   ├── security.py        # Prompt injection protection
│   └── agents/
│       ├── __init__.py
│       ├── base.py        # Base agent class + structured output
│       ├── socratic.py    # Socratic questioner
│       ├── devils_advocate.py
│       ├── clarifier.py
│       ├── synthesizer.py
│       └── perspective.py
├── tests/
│   ├── test_harness.py            # Basic functionality tests
│   ├── test_routing.py            # Routing system tests
│   ├── test_agent_constraints.py  # Agent behavior tests
│   ├── test_session_controls.py   # Session control tests
│   └── test_adaptive_scenarios.py # Comprehensive scenarios
├── pyproject.toml
├── LICENSE
└── README.md
```

## Security Notes

- **API keys**: Never printed or logged. Use config file to avoid shell history leaks.
- **Config files**: Store at `~/.config/dathinker/config.toml` with `chmod 600` permissions.
- **Prompt injection**: Comprehensive protection against manipulation attempts.
- **No secrets in output**: The client repr/str methods mask sensitive data.

## Roadmap

### v0.2 (Current)
- [x] Explainable adaptive routing (trace + pluggable routers)
- [x] Multi-level synthesis (TL;DR + map + next moves)
- [x] Session control surfaces (/pin, /assumptions, /goal, /constraints)
- [x] Structured agent output (intent, targets, proposals)
- [x] Config file support for API keys

### v0.3 (Planned)
- [ ] Export formats: Markdown, JSON, "decision memo"
- [ ] Session persistence (save/load) with redaction options
- [ ] Session history compression

### v0.4 (Planned)
- [ ] Optional retrieval: "bring your own docs" (folder ingest, URLs)
- [ ] Agent citations + "what changed my mind" tracking

## License

MIT - see [LICENSE](LICENSE)
