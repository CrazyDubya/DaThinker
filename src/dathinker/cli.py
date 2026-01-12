"""Command-line interface for DaThinker."""

import asyncio
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

from .openrouter import OpenRouterClient
from .orchestrator import ThinkingOrchestrator, ThinkingMode
from .agents import AgentResponse

# Custom theme for different agents
AGENT_COLORS = {
    "socratic": "cyan",
    "advocate": "red",
    "clarifier": "yellow",
    "synthesizer": "green",
    "expander": "magenta",
}

custom_theme = Theme({
    "agent.socratic": "cyan",
    "agent.advocate": "red",
    "agent.clarifier": "yellow",
    "agent.synthesizer": "green",
    "agent.expander": "magenta",
    "user": "bold white",
    "info": "dim",
    "highlight": "bold yellow",
})

console = Console(theme=custom_theme)


def print_welcome():
    """Print welcome message."""
    welcome = """
# DaThinker

**A multi-agent system that helps you think deeper, not outsource your thinking.**

This tool won't give you answers. Instead, it will:
- Ask probing questions (Socrates)
- Challenge your assumptions (Advocate)
- Clarify vague concepts (Clarifier)
- Find patterns in your thinking (Synthesizer)
- Offer new perspectives (Expander)

**Commands:**
- `/agents` - List available thinking agents
- `/mode <mode>` - Change thinking mode (single, parallel, adaptive)
- `/agent <name>` - Talk to a specific agent
- `/synthesize` - Get a synthesis of your thinking session
- `/summary` - Get session statistics
- `/reset` - Start a new session
- `/help` - Show this help
- `/quit` - Exit

**Tip:** Just type your thoughts and let the agents help you explore them.
"""
    console.print(Panel(Markdown(welcome), title="Welcome", border_style="blue"))


def print_agent_response(response: AgentResponse):
    """Pretty print an agent response."""
    color = AGENT_COLORS.get(response.agent_name.lower(), "white")

    console.print()
    console.print(
        Panel(
            Markdown(response.content),
            title=f"[bold {color}]{response.agent_name}[/]",
            subtitle=f"[dim]{response.role.value}[/]",
            border_style=color,
        )
    )


def print_agents_table(agents: list[str]):
    """Print table of available agents."""
    table = Table(title="Available Thinking Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Description")

    agent_info = {
        "socratic": ("Socratic Questioner", "Asks probing questions to deepen your understanding"),
        "advocate": ("Devil's Advocate", "Challenges your assumptions and presents counterarguments"),
        "clarifier": ("Clarifier", "Identifies ambiguity and helps define terms precisely"),
        "synthesizer": ("Synthesizer", "Finds patterns and organizes your ideas"),
        "expander": ("Perspective Expander", "Offers alternative viewpoints and frames"),
    }

    for name in agents:
        role, desc = agent_info.get(name, (name, ""))
        table.add_row(name, role, desc)

    console.print(table)


async def run_interactive_session():
    """Run an interactive thinking session."""
    print_welcome()

    try:
        client = OpenRouterClient()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Please set OPENROUTER_API_KEY environment variable[/dim]")
        return

    orchestrator = ThinkingOrchestrator(client)

    # Get initial topic
    console.print("\n[bold]What would you like to think about?[/bold]")
    topic = Prompt.ask("[user]Topic[/]")

    if not topic.strip():
        console.print("[dim]No topic provided. Exiting.[/dim]")
        return

    session = orchestrator.start_session(topic, ThinkingMode.ADAPTIVE)
    console.print(f"\n[info]Started thinking session on: {topic}[/info]")
    console.print("[info]Mode: adaptive (agents chosen based on your input)[/info]")

    current_mode = ThinkingMode.ADAPTIVE
    current_agent: str | None = None

    # Callback for streaming agent responses
    async def on_agent_response(response: AgentResponse):
        print_agent_response(response)

    while True:
        try:
            console.print()
            user_input = Prompt.ask("[user]You[/]")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else None

                if cmd == "quit" or cmd == "exit":
                    console.print("\n[info]Thanks for thinking with us![/info]")
                    break

                elif cmd == "help":
                    print_welcome()

                elif cmd == "agents":
                    print_agents_table(orchestrator.list_agents())

                elif cmd == "mode":
                    if arg in ["single", "parallel", "adaptive"]:
                        current_mode = ThinkingMode(arg)
                        console.print(f"[info]Switched to {arg} mode[/info]")
                    else:
                        console.print("[dim]Available modes: single, parallel, adaptive[/dim]")

                elif cmd == "agent":
                    if arg in orchestrator.list_agents():
                        current_agent = arg
                        current_mode = ThinkingMode.SINGLE
                        console.print(f"[info]Now talking to {arg}. Use /mode to change.[/info]")
                    else:
                        console.print(f"[dim]Unknown agent. Available: {', '.join(orchestrator.list_agents())}[/dim]")

                elif cmd == "synthesize":
                    console.print("\n[info]Synthesizing your thinking session...[/info]")
                    synthesis = await orchestrator.synthesize_session()
                    console.print(Panel(
                        Markdown(synthesis),
                        title="Session Synthesis",
                        border_style="blue"
                    ))

                elif cmd == "summary":
                    summary = orchestrator.get_session_summary()
                    console.print(Panel(
                        f"**Session:** {summary.get('id')}\n"
                        f"**Topic:** {summary.get('topic')}\n"
                        f"**Mode:** {summary.get('mode')}\n"
                        f"**Turns:** {summary.get('turns')}\n"
                        f"**Insights:** {summary.get('unique_insights')}\n"
                        f"**Open Questions:** {summary.get('open_questions')}",
                        title="Session Summary",
                        border_style="green"
                    ))

                elif cmd == "reset":
                    console.print("\n[bold]What would you like to think about?[/bold]")
                    topic = Prompt.ask("[user]Topic[/]")
                    if topic.strip():
                        session = orchestrator.start_session(topic, current_mode)
                        console.print(f"[info]Started new session on: {topic}[/info]")

                else:
                    console.print(f"[dim]Unknown command: {cmd}. Type /help for help.[/dim]")

                continue

            # Process thinking input based on mode
            console.print("\n[dim]Thinking...[/dim]")

            if current_mode == ThinkingMode.SINGLE and current_agent:
                response = await orchestrator.think_with_agent(
                    user_input,
                    current_agent,
                )
                print_agent_response(response)

            elif current_mode == ThinkingMode.PARALLEL:
                responses = await orchestrator.think_parallel(user_input)
                for response in responses:
                    print_agent_response(response)

            else:  # ADAPTIVE
                await orchestrator.think_adaptive(
                    user_input,
                    on_agent_response=on_agent_response,
                )

        except KeyboardInterrupt:
            console.print("\n\n[info]Interrupted. Type /quit to exit.[/info]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    asyncio.run(run_interactive_session())


if __name__ == "__main__":
    main()
