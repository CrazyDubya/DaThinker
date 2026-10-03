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
from .router import RouterType, RoutingTrace

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
# DaThinker v0.2

**A multi-agent system that helps you think deeper, not outsource your thinking.**

This tool won't give you answers. Instead, it will:
- Ask probing questions (Socrates)
- Challenge your assumptions (Advocate)
- Clarify vague concepts (Clarifier)
- Find patterns in your thinking (Synthesizer)
- Offer new perspectives (Expander)

**Core Commands:**
- `/agents` - List available thinking agents
- `/mode <mode>` - Change thinking mode (single, parallel, adaptive)
- `/agent <name>` - Talk to a specific agent
- `/synthesize [style]` - Get session synthesis (styles: default, memo, outline, debate, todo)
- `/summary` - Get session statistics
- `/reset` - Start a new session
- `/help` - Show this help
- `/quit` - Exit

**Session Control (v0.2):**
- `/pin <statement>` - Mark as working assumption
- `/assumptions` - Show all pinned assumptions
- `/assume <id> <status>` - Update assumption (open, confirmed, contested, resolved)
- `/goal <statement>` - Set what we're optimizing for
- `/constraint <content>` - Add a constraint (use --category=ethics|time|money|scope|technical)
- `/constraints` - Show all constraints
- `/router [type]` - Show or set router (heuristic, llm, hybrid)
- `/trace` - Show last routing decision details

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


def print_routing_trace(trace: RoutingTrace, verbose: bool = False):
    """Print routing trace information."""
    if verbose:
        console.print(Panel(
            trace.format_detailed(),
            title="Routing Trace",
            border_style="dim",
        ))
    else:
        console.print(f"[dim]{trace.format_summary()}[/dim]")


def print_synthesis(synthesis: dict):
    """Print multi-level synthesis."""
    # TL;DR section
    tldr_text = "\n".join(f"- {item}" for item in synthesis.get("tldr", []))
    console.print(Panel(
        Markdown(f"## TL;DR\n{tldr_text}"),
        title="Summary",
        border_style="blue"
    ))

    # Map section
    map_data = synthesis.get("map", {})
    if map_data:
        map_parts = []
        if map_data.get("key_claims"):
            map_parts.append("**Key Claims:**\n" + "\n".join(f"- {c}" for c in map_data["key_claims"]))
        if map_data.get("evidence"):
            map_parts.append("**Evidence:**\n" + "\n".join(f"- {e}" for e in map_data["evidence"]))
        if map_data.get("assumptions"):
            map_parts.append("**Assumptions:**\n" + "\n".join(f"- {a}" for a in map_data["assumptions"]))
        if map_data.get("open_questions"):
            map_parts.append("**Open Questions:**\n" + "\n".join(f"- {q}" for q in map_data["open_questions"]))
        if map_data.get("conflicts"):
            map_parts.append("**Conflicts:**\n" + "\n".join(f"- {c}" for c in map_data["conflicts"]))

        if map_parts:
            console.print(Panel(
                Markdown("\n\n".join(map_parts)),
                title="Thinking Map",
                border_style="green"
            ))

    # Next moves section
    next_moves = synthesis.get("next_moves", [])
    if next_moves:
        moves_text = []
        for move in next_moves:
            if isinstance(move, dict):
                priority = move.get("priority", "medium")
                priority_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(priority, "white")
                move_type = move.get("type", "action")
                moves_text.append(f"[{priority_color}][{priority.upper()}][/{priority_color}] ({move_type}) {move.get('action', str(move))}")
            else:
                moves_text.append(f"- {move}")

        console.print(Panel(
            "\n".join(moves_text),
            title="Next Moves",
            border_style="magenta"
        ))


def print_assumptions_table(assumptions: list):
    """Print table of assumptions."""
    if not assumptions:
        console.print("[dim]No assumptions pinned yet. Use /pin <statement> to add one.[/dim]")
        return

    table = Table(title="Working Assumptions")
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Status", style="yellow", width=10)
    table.add_column("Content")
    table.add_column("Turn", style="dim", width=5)

    status_colors = {
        "open": "yellow",
        "confirmed": "green",
        "contested": "red",
        "resolved": "dim",
    }

    for a in assumptions:
        color = status_colors.get(a.status, "white")
        table.add_row(
            str(a.id),
            f"[{color}]{a.status}[/{color}]",
            a.content[:60] + "..." if len(a.content) > 60 else a.content,
            str(a.turn_added),
        )

    console.print(table)


def print_constraints_table(constraints: list):
    """Print table of constraints."""
    if not constraints:
        console.print("[dim]No constraints set. Use /constraint <content> to add one.[/dim]")
        return

    table = Table(title="Constraints")
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Category", style="yellow", width=10)
    table.add_column("Content")

    for c in constraints:
        table.add_row(str(c.id), c.category, c.content)

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
    console.print(f"[info]Router: {orchestrator.router_type.value}[/info]")

    current_mode = ThinkingMode.ADAPTIVE
    current_agent: str | None = None
    last_routing_trace: RoutingTrace | None = None
    show_routing_traces = True  # Show routing traces by default

    # Callbacks for adaptive mode
    async def on_agent_response(response: AgentResponse):
        print_agent_response(response)

    async def on_routing_trace(trace: RoutingTrace):
        nonlocal last_routing_trace
        last_routing_trace = trace
        if show_routing_traces:
            print_routing_trace(trace, verbose=False)

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
                    style = arg if arg in ["default", "memo", "outline", "debate", "todo"] else "default"
                    console.print(f"\n[info]Synthesizing your thinking session (style: {style})...[/info]")
                    synthesis = await orchestrator.synthesize_session(style=style)
                    print_synthesis(synthesis)

                elif cmd == "summary":
                    summary = orchestrator.get_session_summary()
                    assumptions = summary.get("assumptions", {})
                    console.print(Panel(
                        f"**Session:** {summary.get('id')}\n"
                        f"**Topic:** {summary.get('topic')}\n"
                        f"**Mode:** {summary.get('mode')}\n"
                        f"**Router:** {summary.get('router')}\n"
                        f"**Goal:** {summary.get('goal')}\n"
                        f"**Turns:** {summary.get('turns')}\n"
                        f"**Insights:** {summary.get('unique_insights')}\n"
                        f"**Open Questions:** {summary.get('open_questions')}\n"
                        f"**Assumptions:** {assumptions.get('total', 0)} total "
                        f"({assumptions.get('open', 0)} open, {assumptions.get('confirmed', 0)} confirmed, {assumptions.get('contested', 0)} contested)\n"
                        f"**Constraints:** {summary.get('constraints')}\n"
                        f"**Routing Decisions:** {summary.get('routing_decisions')}",
                        title="Session Summary",
                        border_style="green"
                    ))

                elif cmd == "reset":
                    console.print("\n[bold]What would you like to think about?[/bold]")
                    topic = Prompt.ask("[user]Topic[/]")
                    if topic.strip():
                        session = orchestrator.start_session(topic, current_mode)
                        last_routing_trace = None
                        console.print(f"[info]Started new session on: {topic}[/info]")

                # v0.2 Session Control Commands
                elif cmd == "pin":
                    if arg:
                        try:
                            assumption = orchestrator.pin_assumption(arg)
                            console.print(f"[green]Pinned assumption #{assumption.id}: {arg}[/green]")
                        except ValueError as e:
                            console.print(f"[red]Error: {e}[/red]")
                    else:
                        console.print("[dim]Usage: /pin <statement to mark as working assumption>[/dim]")

                elif cmd == "assumptions":
                    assumptions = orchestrator.get_assumptions()
                    print_assumptions_table(assumptions)

                elif cmd == "assume":
                    if arg:
                        parts = arg.split(maxsplit=1)
                        if len(parts) >= 2:
                            try:
                                assumption_id = int(parts[0])
                                status = parts[1].lower()
                                if status in ["open", "confirmed", "contested", "resolved"]:
                                    result = orchestrator.update_assumption(assumption_id, status=status)
                                    if result:
                                        console.print(f"[info]Updated assumption #{assumption_id} to {status}[/info]")
                                    else:
                                        console.print(f"[dim]Assumption #{assumption_id} not found[/dim]")
                                else:
                                    console.print("[dim]Valid statuses: open, confirmed, contested, resolved[/dim]")
                            except ValueError:
                                console.print("[dim]Usage: /assume <id> <status>[/dim]")
                        else:
                            console.print("[dim]Usage: /assume <id> <status>[/dim]")
                    else:
                        console.print("[dim]Usage: /assume <id> <status> (e.g., /assume 1 confirmed)[/dim]")

                elif cmd == "goal":
                    if arg:
                        try:
                            orchestrator.set_goal(arg)
                            console.print(f"[green]Goal set: {arg}[/green]")
                        except ValueError as e:
                            console.print(f"[red]Error: {e}[/red]")
                    else:
                        goal = orchestrator.get_goal()
                        if goal:
                            console.print(f"[info]Current goal: {goal}[/info]")
                        else:
                            console.print("[dim]No goal set. Usage: /goal <what we're optimizing for>[/dim]")

                elif cmd == "constraint":
                    if arg:
                        # Parse optional --category flag
                        category = "general"
                        content = arg
                        if "--category=" in arg:
                            parts = arg.split("--category=")
                            content = parts[0].strip()
                            category_part = parts[1].split()[0] if parts[1] else "general"
                            if category_part in ["time", "money", "ethics", "scope", "technical"]:
                                category = category_part
                            # Get remaining content after category
                            remaining = parts[1].split(maxsplit=1)
                            if len(remaining) > 1:
                                content = content + " " + remaining[1] if content else remaining[1]
                        try:
                            constraint = orchestrator.add_constraint(content.strip(), category)
                            console.print(f"[green]Added constraint #{constraint.id} [{category}]: {content.strip()}[/green]")
                        except ValueError as e:
                            console.print(f"[red]Error: {e}[/red]")
                    else:
                        console.print("[dim]Usage: /constraint <content> [--category=time|money|ethics|scope|technical][/dim]")

                elif cmd == "constraints":
                    constraints = orchestrator.get_constraints()
                    print_constraints_table(constraints)

                elif cmd == "router":
                    if arg:
                        try:
                            router_type = RouterType(arg.lower())
                            orchestrator.set_router(router_type)
                            console.print(f"[info]Switched to {router_type.value} router[/info]")
                        except ValueError:
                            console.print("[dim]Available routers: heuristic, llm, hybrid[/dim]")
                    else:
                        router_info = orchestrator.get_router_info()
                        console.print(f"[info]Current router: {router_info['type']} - {router_info['description']}[/info]")

                elif cmd == "trace":
                    if last_routing_trace:
                        print_routing_trace(last_routing_trace, verbose=True)
                    else:
                        console.print("[dim]No routing trace available yet. Send a message in adaptive mode first.[/dim]")

                elif cmd == "traces":
                    # Toggle routing trace display
                    show_routing_traces = not show_routing_traces
                    status = "on" if show_routing_traces else "off"
                    console.print(f"[info]Routing trace display: {status}[/info]")

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
                responses, trace = await orchestrator.think_adaptive(
                    user_input,
                    on_agent_response=on_agent_response,
                    on_routing_trace=on_routing_trace,
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
