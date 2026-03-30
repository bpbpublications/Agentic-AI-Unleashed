# main.py
# ============================================================
# Clinical Research Assistant Agent — Entry Point
# Agentic AI Unleashed — Appendix Demo
#
# Runs an interactive multi-turn conversation with the agent,
# demonstrating all seven concepts in a natural flow:
#
#   Turn 1 → Guidelines query  → RAG + HITL + Memory
#   Turn 2 → Drug info query   → Drug Agent + HITL
#   Turn 3 → Follow-up         → Memory (references Turn 1)
#   Turn 4 → Out-of-scope      → Direct response
#
# Usage:
#   # Terminal 1: start the MCP server
#   python mcp_server.py
#
#   # Terminal 2: run the demo
#   python main.py
#
# To run interactively (enter your own queries):
#   python main.py --interactive
# ============================================================
from dotenv import load_dotenv
load_dotenv()

import sys
import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from agents.graph import clinical_graph
from utils.config import setup

console = Console()
config = setup()


# ── Display Helpers ───────────────────────────────────────────

def print_user_query(query: str):
    console.print()
    console.print(Rule("[bold cyan]User Query[/bold cyan]", style="cyan"))
    console.print(Panel(query, style="cyan", expand=False))

def print_agent_response(response: str):
    console.print(Rule("[bold green]Agent Response[/bold green]", style="green"))
    console.print(Panel(response, style="green"))

def print_hitl_prompt(interrupt_data: dict):
    console.print()
    console.print(Rule("[bold yellow]⚠ Human Review Required[/bold yellow]", style="yellow"))
    console.print(Panel(
        f"[bold]Draft Response:[/bold]\n{interrupt_data.get('draft_response', '')}\n\n"
        f"[bold]Evidence Sources:[/bold]\n" +
        "\n".join(f"  • {s}" for s in interrupt_data.get('evidence_sources', [])),
        title="Review Before Delivery",
        style="yellow",
    ))

def print_routing(message: str):
    console.print(f"[dim]  → {message}[/dim]")


# ── Core Invocation Logic ─────────────────────────────────────

def run_turn(
    query: str,
    thread_id: str,
    auto_approve: bool = True,
    human_input: str | None = None,
) -> str:
    """
    Run a single conversation turn through the graph.

    Handles the two-phase invocation pattern required for HITL:
      Phase 1: Initial invoke — may pause at human_review node
      Phase 2: Resume with human decision (approve / modify)

    Args:
        query:        The user's message.
        thread_id:    Conversation thread identifier (enables memory).
        auto_approve: If True, auto-approves HITL in demo mode.
        human_input:  Override the HITL decision (for demo scripting).
    """
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Phase 1: Initial invocation
    result = clinical_graph.invoke(
        {"messages": [HumanMessage(content=query)],
         "requires_review": False,
         "human_approved": False,
         "specialist_response": None,
         "routing_decision": ""},
        config=thread_config,
    )

    # Check if graph paused at a HITL interrupt
    # LangGraph surfaces interrupts via __interrupt__ in the state
    interrupt_data = None
    if hasattr(result, "__interrupt__") or isinstance(result, dict) and "__interrupt__" in result:
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            interrupt_data = interrupts[0].value if hasattr(interrupts[0], 'value') else interrupts[0]

    if interrupt_data:
        print_hitl_prompt(interrupt_data)

        # Determine reviewer decision
        if auto_approve:
            decision = "approve"
            console.print("[yellow]  [Auto-demo] Approving response...[/yellow]")
        elif human_input:
            decision = human_input
        else:
            decision = console.input("\n[yellow]  Your decision (approve / type correction): [/yellow]")

        # Phase 2: Resume graph with human decision
        result = clinical_graph.invoke(
            Command(resume=decision),
            config=thread_config,
        )

    # Extract the final response from the last AI message
    messages = result.get("messages", [])
    ai_messages = [m for m in messages if hasattr(m, 'content') and not isinstance(m, HumanMessage)]

    # Return the last substantive response (not routing messages)
    for msg in reversed(ai_messages):
        content = msg.content
        if content and not content.startswith("[Routing"):
            return content

    return "No response generated."


# ── Scripted Demo Conversation ────────────────────────────────

DEMO_TURNS = [
    {
        "label": "Turn 1 — Clinical Guidelines Query",
        "query": "What are the current USPSTF recommendations for hypertension screening?",
        "note":  "Demonstrates: RAG retrieval, Guidelines Agent, HITL review, Memory initialization",
    },
    {
        "label": "Turn 2 — Drug Information Query",
        "query": "What are the contraindications and key drug interactions for lisinopril?",
        "note":  "Demonstrates: Drug Agent routing, MCP tool call, Structured Output (DrugInfoResponse), HITL",
    },
    {
        "label": "Turn 3 — Follow-up with Memory",
        "query": "Given the hypertension guidelines you just described, when would lisinopril be a first-line choice?",
        "note":  "Demonstrates: Memory — agent references context from Turn 1 and Turn 2",
    },
    {
        "label": "Turn 4 — Out-of-Scope (Direct Response)",
        "query": "Thank you, that was very helpful. What is this system built with?",
        "note":  "Demonstrates: Supervisor routing to direct_response, no tool calls",
    },
]


def run_demo(interactive: bool = False):
    """
    Run the full demo conversation.
    In demo mode: scripted turns with auto-approval.
    In interactive mode: user types queries, HITL prompts are real.
    """
    thread_id = str(uuid.uuid4())    # Unique thread for this session

    console.print()
    console.print(Panel(
        "[bold white]Clinical Research Assistant Agent[/bold white]\n"
        "[dim]Agentic AI Unleashed — Appendix Demo[/dim]\n\n"
        f"[dim]Thread ID: {thread_id}[/dim]\n"
        f"[dim]LangSmith: {'enabled' if config['langsmith_enabled'] else 'disabled'}[/dim]",
        title="🏥 Demo Starting",
        style="white",
        expand=False,
    ))

    if interactive:
        _run_interactive(thread_id)
    else:
        _run_scripted(thread_id)


def _run_scripted(thread_id: str):
    """Run the pre-scripted demo turns."""
    for i, turn in enumerate(DEMO_TURNS, 1):
        console.print(f"\n[bold white]{'═' * 60}[/bold white]")
        console.print(f"[bold white]{turn['label']}[/bold white]")
        console.print(f"[dim italic]{turn['note']}[/dim italic]")

        print_user_query(turn["query"])

        response = run_turn(
            query=turn["query"],
            thread_id=thread_id,
            auto_approve=True,
        )

        print_agent_response(response)

    console.print(f"\n[bold green]✓ Demo complete.[/bold green]")
    if config["langsmith_enabled"]:
        console.print(
            f"[dim]View full traces at: https://smith.langchain.com "
            f"→ Project: {config['langsmith_project']}[/dim]"
        )


def _run_interactive(thread_id: str):
    """Run in interactive mode — user enters queries manually."""
    console.print("\n[cyan]Interactive mode. Type 'quit' to exit.[/cyan]\n")

    while True:
        query = console.input("[cyan]You: [/cyan]").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        response = run_turn(
            query=query,
            thread_id=thread_id,
            auto_approve=False,      # Real HITL prompts in interactive mode
        )
        print_agent_response(response)


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    interactive = "--interactive" in sys.argv
    run_demo(interactive=interactive)
