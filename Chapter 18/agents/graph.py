# agents/graph.py
# ============================================================
# LangGraph Multi-Agent Supervisor Graph
# Agentic AI Unleashed — Appendix Demo
#
# This is the core of the demonstration. It implements a
# supervisor / specialist agent architecture using LangGraph,
# combining all seven concepts in a single coherent graph:
#
#   1. Multi-Agent routing (Supervisor → Specialist)
#   2. MCP Tool calls (via HTTP to FastAPI server)
#   3. RAG (tools invoke FAISS retrieval under the hood)
#   4. Structured Output (Pydantic models throughout)
#   5. Memory (MemorySaver checkpointer)
#   6. Human-in-the-Loop (interrupt before final delivery)
#   7. Observability (LangSmith traces every node automatically)
#
# ── Agent-to-Agent (A2A) Note ────────────────────────────────
# The supervisor routes to specialist agents by calling their
# node functions directly within the LangGraph process.
# In a production A2A deployment (Google A2A spec), each
# specialist would be an independent service. The supervisor
# would replace the direct node call with an HTTP POST to the
# specialist's A2A endpoint, e.g.:
#
#   POST https://drug-agent.internal/a2a/tasks/send
#   { "message": { "role": "user", "parts": [{"text": query}] } }
#
# The graph structure and routing logic would remain identical.
# ============================================================

from typing import Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from agents.models import ClinicalResponse, DrugInfoResponse, SupervisorDecision
from agents.tools import GUIDELINES_TOOLS, DRUG_INFO_TOOLS
from utils.config import setup

config = setup()


# ── State Definition ──────────────────────────────────────────
# The shared state flows through every node in the graph.
# 'messages' uses the add_messages reducer so each node appends
# rather than overwrites — giving us a full conversation history.

class AgentState(TypedDict):
    messages:         Annotated[list[BaseMessage], add_messages]
    routing_decision: str                  # Which specialist was chosen
    specialist_response: dict | None       # Structured output from specialist
    requires_review:  bool                 # Whether HITL checkpoint is triggered
    human_approved:   bool                 # Set to True after human reviews


# ── LLM Initialisation ────────────────────────────────────────

llm = ChatOpenAI(
    model=config["openai_model"],
    temperature=0,                         # Deterministic for clinical context
)


# ════════════════════════════════════════════════════════════
# NODE 1: SUPERVISOR
# Decides how to route the user's query.
# ════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM = """You are a clinical research supervisor agent. Your role is to
analyze the user's question and route it to the most appropriate specialist:

- guidelines_agent: For questions about clinical recommendations, screening guidelines,
  treatment thresholds, preventive care, or evidence-based practice protocols.
- drug_agent: For questions about specific medications — indications, contraindications,
  drug interactions, dosing, or safety warnings.
- respond_directly: For greetings, clarifications, or questions clearly outside the
  clinical domain.

Be precise and decisive. Analyze the question carefully before routing."""

def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor Agent: Analyzes the incoming query and decides routing.
    Uses structured output (SupervisorDecision) to make routing explicit
    and inspectable in LangSmith traces.
    """
    print("\n[Supervisor] Analyzing query and determining route...")

    structured_llm = llm.with_structured_output(SupervisorDecision)

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        *state["messages"],
    ]

    decision: SupervisorDecision = structured_llm.invoke(messages)
    print(f"[Supervisor] Routing to: {decision.route_to} — {decision.reasoning}")

    return {
        **state,
        "routing_decision": decision.route_to,
        "messages": state["messages"] + [
            AIMessage(content=f"[Routing to {decision.route_to}]: {decision.reasoning}")
        ],
    }


# ── Routing Function (Edge Condition) ────────────────────────

def route_query(state: AgentState) -> Literal["guidelines_agent", "drug_agent", "direct_response"]:
    """
    LangGraph conditional edge: reads the supervisor's routing decision
    and directs the graph to the appropriate specialist node.

    # A2A Extension Point:
    # In a multi-service deployment, this function would look up
    # the A2A endpoint URL for each specialist and return it,
    # allowing the framework to dispatch via HTTP rather than
    # local function call.
    """
    route = state.get("routing_decision", "direct_response")
    if route == "guidelines_agent":
        return "guidelines_agent"
    elif route == "drug_agent":
        return "drug_agent"
    else:
        return "direct_response"


# ════════════════════════════════════════════════════════════
# NODE 2: GUIDELINES AGENT (Specialist)
# Handles clinical guideline questions using RAG via MCP tools.
# ════════════════════════════════════════════════════════════

GUIDELINES_SYSTEM = """You are a clinical guidelines specialist. Your task is to answer
clinical questions accurately using only the retrieved evidence provided by your tools.

Always:
- Call the search_guidelines tool before answering
- Base your response strictly on retrieved evidence
- Cite the source titles in your response
- Be explicit about the strength of evidence (Grade A, B, etc. where available)
- Set requires_human_review=True if the question involves treatment decisions
  for specific patients or high-stakes clinical scenarios"""

def guidelines_agent_node(state: AgentState) -> AgentState:
    """
    Guidelines Agent: A specialist that answers clinical guideline questions.
    It binds the search_guidelines MCP tool and uses it to retrieve
    evidence before formulating a structured response.

    # A2A Note: In a distributed deployment, this entire node would
    # be replaced by an HTTP call to a standalone Guidelines Agent
    # service that implements the A2A task protocol. The supervisor
    # would POST a task and await the structured response.
    """
    print("[Guidelines Agent] Retrieving relevant clinical guidelines...")

    # Bind the MCP tool to the LLM
    agent_llm = llm.bind_tools(GUIDELINES_TOOLS)
    structured_llm = llm.with_structured_output(ClinicalResponse)

    # Step 1: Let the agent call the tool
    tool_response = agent_llm.invoke([
        SystemMessage(content=GUIDELINES_SYSTEM),
        *state["messages"],
    ])

    # Step 2: Execute the tool calls if any were requested
    tool_results = []
    if tool_response.tool_calls:
        for tool_call in tool_response.tool_calls:
            if tool_call["name"] == "search_guidelines":
                result = GUIDELINES_TOOLS[0].invoke(tool_call["args"])
                tool_results.append(result)
                print(f"[Guidelines Agent] Retrieved evidence from MCP server.")

    # Step 3: Generate structured final response with evidence context
    evidence_context = "\n\n".join(tool_results) if tool_results else "No specific guidelines retrieved."

    final_response: ClinicalResponse = structured_llm.invoke([
        SystemMessage(content=GUIDELINES_SYSTEM),
        *state["messages"],
        HumanMessage(content=f"Based on this retrieved evidence, provide your structured response:\n\n{evidence_context}"),
    ])

    print(f"[Guidelines Agent] Response ready. Confidence: {final_response.confidence_level}. "
          f"Requires review: {final_response.requires_human_review}")

    return {
        **state,
        "specialist_response": final_response.model_dump(),
        "requires_review": final_response.requires_human_review,
        "messages": state["messages"] + [
            AIMessage(content=final_response.recommendation)
        ],
    }


# ════════════════════════════════════════════════════════════
# NODE 3: DRUG INFORMATION AGENT (Specialist)
# Handles pharmacology questions using MCP drug info tool.
# ════════════════════════════════════════════════════════════

DRUG_AGENT_SYSTEM = """You are a clinical pharmacology specialist. Your task is to provide
accurate drug information using only the retrieved FDA DailyMed-sourced data.

Always:
- Call the get_drug_info tool before answering
- Extract the drug name precisely from the user's question
- Include contraindications and safety warnings in your response
- Note significant drug interactions
- Set evidence_sources to the retrieved document titles"""

def drug_agent_node(state: AgentState) -> AgentState:
    """
    Drug Information Agent: A specialist that answers pharmacology questions.
    Binds the get_drug_info MCP tool for evidence retrieval.

    # A2A Note: Same as the Guidelines Agent — this node maps directly
    # to an independent A2A-compliant Drug Information Agent service.
    """
    print("[Drug Agent] Retrieving drug information from MCP server...")

    agent_llm = llm.bind_tools(DRUG_INFO_TOOLS)
    structured_llm = llm.with_structured_output(DrugInfoResponse)

    # Step 1: Tool call
    tool_response = agent_llm.invoke([
        SystemMessage(content=DRUG_AGENT_SYSTEM),
        *state["messages"],
    ])

    # Step 2: Execute tool calls
    tool_results = []
    if tool_response.tool_calls:
        for tool_call in tool_response.tool_calls:
            if tool_call["name"] == "get_drug_info":
                result = DRUG_INFO_TOOLS[0].invoke(tool_call["args"])
                tool_results.append(result)
                print(f"[Drug Agent] Retrieved drug data from MCP server.")

    # Step 3: Check for empty / not-found results before invoking the LLM
    evidence_context = "\n\n".join(tool_results) if tool_results else ""
    no_results = not evidence_context.strip() or evidence_context.strip() in (
        "No drug information found", "No drug information retrieved."
    )

    if no_results:
        print("[Drug Agent] No drug data found in corpus — returning not-found response.")
        not_found_message = (
            "I'm sorry, I don't have information about that drug in my current corpus. "
            "This assistant only covers Metformin, Lisinopril, and Atorvastatin. "
            "For other medications, please consult the FDA DailyMed database at "
            "https://dailymed.nlm.nih.gov/dailymed/."
        )
        return {
            **state,
            "specialist_response": None,
            "requires_review": False,      # No clinical content to review
            "messages": state["messages"] + [AIMessage(content=not_found_message)],
        }

    # Step 4: Structured response (only reached when evidence was found)
    final_response: DrugInfoResponse = structured_llm.invoke([
        SystemMessage(content=DRUG_AGENT_SYSTEM),
        *state["messages"],
        HumanMessage(content=f"Based on this retrieved drug information, provide your structured response:\n\n{evidence_context}"),
    ])

    print(f"[Drug Agent] Drug info response ready for: {final_response.drug_name}")

    return {
        **state,
        "specialist_response": final_response.model_dump(),
        "requires_review": True,           # Drug info always warrants human review
        "messages": state["messages"] + [
            AIMessage(content=f"{final_response.indication}\n\nSafety Note: {final_response.safety_note}")
        ],
    }


# ════════════════════════════════════════════════════════════
# NODE 4: DIRECT RESPONSE
# Handles greetings and out-of-scope queries without tools.
# ════════════════════════════════════════════════════════════

def direct_response_node(state: AgentState) -> AgentState:
    """
    Direct Response Node: Handles non-clinical queries conversationally.
    No tool calls, no specialist routing — just the LLM responding directly.
    """
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful clinical research assistant. "
            "Respond helpfully to the user's message. "
            "For clinical questions, let them know you can help with guidelines and drug information."
        )),
        *state["messages"],
    ])

    return {
        **state,
        "specialist_response": None,
        "requires_review": False,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


# ════════════════════════════════════════════════════════════
# NODE 5: HUMAN-IN-THE-LOOP REVIEW
# Pauses the graph and surfaces the response for human approval
# before it is delivered to the user.
# ════════════════════════════════════════════════════════════

def human_review_node(state: AgentState) -> AgentState:
    """
    Human-in-the-Loop (HITL) Node.

    This node uses LangGraph's interrupt() to pause graph execution
    and surface the agent's draft response to a human reviewer.
    The graph will resume only after the human calls graph.invoke()
    again with the same thread_id and their approval decision.

    Why HITL for clinical responses?
    Clinical recommendations can directly influence patient care.
    Requiring a human to confirm before delivery is not just a
    technical pattern — it reflects responsible AI deployment.
    The interrupt here makes that intent explicit and enforceable.

    In the demo, we simulate the human reviewer in main.py.
    In production, this interrupt would surface a notification in
    a clinical dashboard, Slack alert, or review queue.
    """
    specialist_response = state.get("specialist_response", {})
    recommendation = specialist_response.get("recommendation") or \
                     specialist_response.get("indication", "Response not available.")
    sources = specialist_response.get("evidence_sources", [])

    # interrupt() pauses the graph here and yields control back to the caller.
    # The caller (main.py) receives the interrupt value, displays it to the user,
    # and must re-invoke the graph to continue.
    human_decision = interrupt({
        "message": "Human review required before delivering this clinical response.",
        "draft_response": recommendation,
        "evidence_sources": sources,
        "instruction": "Review the draft response above. Type 'approve' to deliver it, or provide corrections.",
    })

    # When the graph resumes, human_decision contains the reviewer's input
    approved = str(human_decision).strip().lower() == "approve"
    print(f"\n[HITL] Human decision: {'APPROVED' if approved else 'REJECTED/MODIFIED'}")

    return {
        **state,
        "human_approved": approved,
    }


# ── HITL Routing ─────────────────────────────────────────────

def should_review(state: AgentState) -> Literal["human_review", "format_response"]:
    """Conditional edge: route to HITL if the specialist flagged review needed."""
    if state.get("requires_review", False):
        return "human_review"
    return "format_response"


# ════════════════════════════════════════════════════════════
# NODE 6: FORMAT RESPONSE
# Assembles the final response shown to the user.
# ════════════════════════════════════════════════════════════

def format_response_node(state: AgentState) -> AgentState:
    """
    Final formatting node: assembles a clean, user-facing response
    from the specialist's structured output.
    Adds a disclaimer appropriate for a demo clinical context.
    """
    specialist_response = state.get("specialist_response")

    if not specialist_response:
        # Direct response path — last message is already the response
        return state

    # Build a well-structured final answer from the Pydantic model data
    sources_text = "\n".join(
        f"  • {s}" for s in specialist_response.get("evidence_sources", [])
    )

    # Handle both ClinicalResponse and DrugInfoResponse shapes
    if "recommendation" in specialist_response:
        body = specialist_response["recommendation"]
        confidence = specialist_response.get("confidence_level", "unknown")
        footer = f"\n\nConfidence: {confidence.upper()} | Sources:\n{sources_text}"
    else:
        body = (
            f"Drug: {specialist_response.get('drug_name', 'Unknown')}\n\n"
            f"Indication: {specialist_response.get('indication', '')}\n\n"
            f"Safety Note: {specialist_response.get('safety_note', '')}\n\n"
            f"Key Interactions: {', '.join(specialist_response.get('key_interactions', []))}"
        )
        footer = f"\n\nSources:\n{sources_text}"

    disclaimer = (
        "\n\n─────────────────────────────────────────────\n"
        "⚠ DEMO ONLY: This information is for educational purposes.\n"
        "It is not a substitute for professional clinical judgment."
    )

    final_text = body + footer + disclaimer

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=final_text)],
    }


# ════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ════════════════════════════════════════════════════════════

def build_graph():
    """
    Assemble and compile the LangGraph multi-agent graph.

    Graph topology:
      START
        └─► supervisor
              ├─► guidelines_agent  ──┐
              ├─► drug_agent         ─┼─► [should_review?]
              └─► direct_response   ──┘        │
                                          ┌────┴─────┐
                                     human_review  format_response
                                          │              │
                                     format_response    END
                                          │
                                         END

    Memory: MemorySaver checkpoints the full state after every node,
    enabling multi-turn conversations within the same thread_id.

    LangSmith: Every node execution is automatically traced when
    LANGCHAIN_TRACING_V2=true is set in the environment.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("guidelines_agent", guidelines_agent_node)
    graph.add_node("drug_agent",       drug_agent_node)
    graph.add_node("direct_response",  direct_response_node)
    graph.add_node("human_review",     human_review_node)
    graph.add_node("format_response",  format_response_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor routes to specialists via conditional edge
    graph.add_conditional_edges(
        "supervisor",
        route_query,
        {
            "guidelines_agent": "guidelines_agent",
            "drug_agent":       "drug_agent",
            "direct_response":  "direct_response",
        },
    )

    # After each specialist, decide whether HITL review is needed
    graph.add_conditional_edges("guidelines_agent", should_review)
    graph.add_conditional_edges("drug_agent",       should_review)
    graph.add_edge("direct_response", "format_response")

    # After HITL review, always format (whether approved or not)
    graph.add_edge("human_review",    "format_response")
    graph.add_edge("format_response", END)

    # Compile with MemorySaver for persistent multi-turn memory
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    print("[Graph] Multi-agent graph compiled with MemorySaver checkpointer.")
    return compiled


# Module-level compiled graph instance
clinical_graph = build_graph()
