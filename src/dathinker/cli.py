"""Command-line interface for DaThinker."""

import asyncio
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme
from rich.tree import Tree

from .openrouter import OpenRouterClient
from .orchestrator import (
    ThinkingOrchestrator,
    ThinkingMode,
    SynthesisStyle,
    AssumptionStatus,
    MultiLevelSynthesis,
)
from .agents import AgentResponse
from .router import RouterVersion, RoutingTrace

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
    "routing": "dim cyan",
    "pin": "bold blue",
    "goal": "bold green",
    "constraint": "bold red",
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
- `/router <type>` - Change routing strategy (heuristic, llm, hybrid)
- `/routing` - Show last routing decision trace

**Session Control:**
- `/pin <statement>` - Mark a statement as working assumption
- `/assumptions` - Show all pinned assumptions
- `/goal <goal>` - Set a session goal
- `/goals` - Show session goals
- `/constraint <constraint>` - Add a constraint agents must respect
- `/constraints` - Show constraints

**Synthesis:**
- `/synthesize [style]` - Get synthesis (memo|outline|debate|todo)
- `/nextstep` - Get smallest next step to reduce uncertainty
- `/summary` - Get session statistics

**Other:**
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


def print_routing_trace(trace: RoutingTrace):
    """Pretty print a routing trace for explainability."""
    console.print()

    # Create a tree for routing visualization
    tree = Tree(f"[bold cyan]Routing Decision[/] [dim](router: {trace.router_version.value})[/]")

    # Add considered agents with scores
    agents_branch = tree.add("[bold]Agents Considered[/]")
    sorted_agents = sorted(trace.considered_agents, key=lambda x: x.score, reverse=True)
    for agent in sorted_agents:
        selected = agent.agent_name in trace.selected_agents
        marker = "[green]>[/] " if selected else "  "
        score_color = "green" if agent.score > 0.5 else "yellow" if agent.score > 0.2 else "dim"
        agents_branch.add(
            f"{marker}[bold]{agent.agent_name}[/]: [{score_color}]{agent.score:.2f}[/] - {agent.rationale}"
        )

    # Add selection info
    selection_branch = tree.add("[bold]Selection[/]")
    selection_branch.add(f"Selected: [bold]{', '.join(trace.selected_agents)}[/]")
    selection_branch.add(f"Confidence: {trace.confidence:.2f}")
    if trace.tie_break_used:
        selection_branch.add(f"Tie-break: {trace.tie_break_method}")

    # Add rationale
    tree.add(f"[dim]Rationale: {trace.selection_rationale}[/]")

    console.print(Panel(tree, title="[routing]Routing Trace[/]", border_style="dim"))


def print_synthesis(synthesis: MultiLevelSynthesis):
    """Pretty print a multi-level synthesis."""
    console.print()

    # TL;DR
    if synthesis.tldr:
        tldr_text = "\n".join([f"- {item}" for item in synthesis.tldr])
        console.print(Panel(Markdown(tldr_text), title="[bold]TL;DR[/]", border_style="blue"))

    # Create a table for the map
    if any([synthesis.key_claims, synthesis.evidence, synthesis.assumptions,
            synthesis.open_questions, synthesis.conflicts]):
        table = Table(title="Thinking Map", show_header=True, header_style="bold")
        table.add_column("Category", style="cyan", width=15)
        table.add_column("Items", style="white")

        if synthesis.key_claims:
            table.add_row("Key Claims", "\n".join([f"- {c}" for c in synthesis.key_claims]))
        if synthesis.evidence:
            table.add_row("Evidence", "\n".join([f"- {e}" for e in synthesis.evidence]))
        if synthesis.assumptions:
            table.add_row("Assumptions", "\n".join([f"- {a}" for a in synthesis.assumptions]))
        if synthesis.open_questions:
            table.add_row("Open Questions", "\n".join([f"- {q}" for q in synthesis.open_questions]))
        if synthesis.conflicts:
            table.add_row("Conflicts", "\n".join([f"- {c}" for c in synthesis.conflicts]))

        console.print(table)

    # Next moves
    if synthesis.next_moves:
        moves_text = "\n".join([f"- {move}" for move in synthesis.next_moves])
        console.print(Panel(Markdown(moves_text), title="[bold green]Next Moves[/]", border_style="green"))


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


def print_assumptions(orchestrator: ThinkingOrchestrator):
    """Print all pinned assumptions."""
    pins = orchestrator.get_pins()

    if not pins:
        console.print("[dim]No assumptions pinned yet. Use /pin <statement> to add one.[/dim]")
        return

    table = Table(title="Working Assumptions")
    table.add_column("ID", style="dim")
    table.add_column("Status", style="cyan")
    table.add_column("Assumption")
    table.add_column("Turn", style="dim")

    status_colors = {
        AssumptionStatus.OPEN: "yellow",
        AssumptionStatus.CONFIRMED: "green",
        AssumptionStatus.CONTESTED: "red",
        AssumptionStatus.REVISED: "blue",
    }

    for pin in pins:
        color = status_colors.get(pin.status, "white")
        table.add_row(
            pin.id,
            f"[{color}]{pin.status.value}[/]",
            pin.content,
            str(pin.turn_created),
        )

    console.print(table)


def print_goals(orchestrator: ThinkingOrchestrator):
    """Print session goals."""
    goals = orchestrator.get_goals(active_only=False)

    if not goals:
        console.print("[dim]No goals set. Use /goal <goal> to add one.[/dim]")
        return

    table = Table(title="Session Goals")
    table.add_column("ID", style="dim")
    table.add_column("Priority", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Goal")

    for goal in goals:
        status = "[green]Active[/]" if goal.active else "[dim]Completed[/]"
        table.add_row(goal.id, str(goal.priority), status, goal.content)

    console.print(table)


def print_constraints(orchestrator: ThinkingOrchestrator):
    """Print session constraints."""
    constraints = orchestrator.get_constraints()

    if not constraints:
        console.print("[dim]No constraints set. Use /constraint <constraint> to add one.[/dim]")
        return

    table = Table(title="Session Constraints")
    table.add_column("ID", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Constraint")

    for c in constraints:
        ctype = "[red]HARD[/]" if c.hard else "[yellow]SOFT[/]"
        table.add_row(c.id, ctype, c.content)

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

    orchestrator = ThinkingOrchestrator(client, router_version=RouterVersion.HYBRID)

    # Get initial topic
    console.print("\n[bold]What would you like to think about?[/bold]")
    topic = Prompt.ask("[user]Topic[/]")

    if not topic.strip():
        console.print("[dim]No topic provided. Exiting.[/dim]")
        return

    session = orchestrator.start_session(topic, ThinkingMode.ADAPTIVE)
    console.print(f"\n[info]Started thinking session on: {topic}[/info]")
    console.print(f"[info]Mode: adaptive | Router: {orchestrator.router_version.value}[/info]")

    current_mode = ThinkingMode.ADAPTIVE
    current_agent: str | None = None
    show_routing = True  # Show routing traces by default

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

                elif cmd == "router":
                    if arg in ["heuristic", "llm", "hybrid"]:
                        version = RouterVersion(arg)
                        orchestrator.set_router(version)
                        console.print(f"[info]Switched to {arg} router[/info]")
                    else:
                        console.print(f"[dim]Available routers: heuristic, llm, hybrid (current: {orchestrator.router_version.value})[/dim]")

                elif cmd == "routing":
                    trace = orchestrator.get_last_routing_trace()
                    if trace:
                        print_routing_trace(trace)
                    else:
                        console.print("[dim]No routing trace yet. Send a message in adaptive mode first.[/dim]")

                elif cmd == "pin":
                    if arg:
                        pin = orchestrator.pin(arg)
                        console.print(f"[pin]Pinned:[/pin] {pin.content} [dim](id: {pin.id})[/dim]")
                    else:
                        console.print("[dim]Usage: /pin <statement to pin as assumption>[/dim]")

                elif cmd == "assumptions":
                    print_assumptions(orchestrator)

                elif cmd == "goal":
                    if arg:
                        # Check for priority suffix like "2:"
                        priority = 1
                        content = arg
                        if arg[0].isdigit() and ":" in arg:
                            parts = arg.split(":", 1)
                            try:
                                priority = int(parts[0])
                                content = parts[1].strip()
                            except ValueError:
                                pass
                        goal = orchestrator.add_goal(content, priority)
                        console.print(f"[goal]Goal added:[/goal] {goal.content} [dim](priority: {goal.priority})[/dim]")
                    else:
                        console.print("[dim]Usage: /goal <goal> or /goal 2: <goal with priority 2>[/dim]")

                elif cmd == "goals":
                    print_goals(orchestrator)

                elif cmd == "constraint":
                    if arg:
                        # Check for soft prefix
                        hard = True
                        content = arg
                        if arg.lower().startswith("soft:"):
                            hard = False
                            content = arg[5:].strip()
                        constraint = orchestrator.add_constraint(content, hard)
                        ctype = "HARD" if hard else "SOFT"
                        console.print(f"[constraint]Constraint added ({ctype}):[/constraint] {constraint.content}")
                    else:
                        console.print("[dim]Usage: /constraint <rule> or /constraint soft: <soft rule>[/dim]")

                elif cmd == "constraints":
                    print_constraints(orchestrator)

                elif cmd == "synthesize":
                    style = SynthesisStyle.MEMO
                    if arg in ["memo", "outline", "debate", "todo"]:
                        style = SynthesisStyle(arg)
                    console.print(f"\n[info]Synthesizing your thinking session (style: {style.value})...[/info]")
                    synthesis = await orchestrator.synthesize_session(style)
                    print_synthesis(synthesis)

                elif cmd == "nextstep":
                    console.print("\n[info]Finding the smallest step to reduce uncertainty...[/info]")
                    next_step = await orchestrator.get_smallest_uncertainty_reducer()
                    console.print(Panel(
                        Markdown(next_step),
                        title="[bold green]Smallest Next Step[/]",
                        border_style="green"
                    ))

                elif cmd == "summary":
                    summary = orchestrator.get_session_summary()
                    summary_text = f"""**Session:** {summary.get('id')}
**Topic:** {summary.get('topic')}
**Mode:** {summary.get('mode')}
**Router:** {summary.get('router')}
**Turns:** {summary.get('turns')}
**Insights:** {summary.get('unique_insights')}
**Open Questions:** {summary.get('open_questions')}
**Pinned Assumptions:** {summary.get('pins')}
**Active Goals:** {summary.get('goals')}
**Constraints:** {summary.get('constraints')}
**Routing Traces:** {summary.get('routing_traces')}"""
                    console.print(Panel(
                        Markdown(summary_text),
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
                responses, routing_trace = await orchestrator.think_adaptive(
                    user_input,
                    on_agent_response=on_agent_response,
                )
                # Show routing trace after responses
                if show_routing:
                    print_routing_trace(routing_trace)

        except KeyboardInterrupt:
            console.print("\n\n[info]Interrupted. Type /quit to exit.[/info]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    asyncio.run(run_interactive_session())


if __name__ == "__main__":
    main()
