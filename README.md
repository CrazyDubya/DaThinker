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

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DaThinker.git
cd DaThinker

# Install dependencies
pip install -e .

# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"
```

## Usage

### Interactive CLI

```bash
# Start an interactive thinking session
dathinker
```

### Commands

- `/agents` - List available thinking agents
- `/mode <mode>` - Change thinking mode (single, parallel, adaptive)
- `/agent <name>` - Talk to a specific agent
- `/synthesize` - Get a synthesis of your thinking session
- `/summary` - Get session statistics
- `/reset` - Start a new session
- `/help` - Show help
- `/quit` - Exit

### Programmatic Usage

```python
import asyncio
from dathinker.orchestrator import ThinkingOrchestrator, ThinkingMode

async def main():
    orchestrator = ThinkingOrchestrator()
    orchestrator.start_session("Career decisions", ThinkingMode.ADAPTIVE)

    responses = await orchestrator.think_adaptive(
        "I'm thinking about changing careers to something more meaningful."
    )

    for response in responses:
        print(f"{response.agent_name}: {response.content}")

    # Get session synthesis
    synthesis = await orchestrator.synthesize_session()
    print(synthesis)

asyncio.run(main())
```

## Running Tests

```bash
python tests/test_harness.py
```

## Models Used

DaThinker uses cost-effective models via OpenRouter:

- **Fast**: `meta-llama/llama-3.1-8b-instruct:free` - For quick meta-decisions
- **Balanced**: `google/gemini-2.0-flash-001` - Default for agents
- **Reasoning**: `deepseek/deepseek-chat` - For complex reasoning tasks

## Project Structure

```
DaThinker/
├── src/dathinker/
│   ├── __init__.py
│   ├── openrouter.py      # OpenRouter API client
│   ├── orchestrator.py    # Multi-agent orchestration
│   ├── cli.py             # Interactive CLI
│   └── agents/
│       ├── __init__.py
│       ├── base.py        # Base agent class
│       ├── socratic.py    # Socratic questioner
│       ├── devils_advocate.py
│       ├── clarifier.py
│       ├── synthesizer.py
│       └── perspective.py
├── tests/
│   └── test_harness.py
├── pyproject.toml
└── README.md
```

## License

MIT
