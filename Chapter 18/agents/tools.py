# agents/tools.py
# ============================================================
# LangChain Tool Wrappers for MCP Server
# Agentic AI Unleashed — Appendix Demo
#
# These LangChain tools form the bridge between the LangGraph
# agents and the MCP tool server. Each tool makes an HTTP POST
# to the corresponding MCP endpoint and returns the result as
# a formatted string for the LLM to reason over.
#
# This separation is architecturally significant:
#   - The agents are completely decoupled from retrieval logic
#   - Tools could be swapped for real MCP SDK tools with minimal
#     changes to the agent graph
#   - The HTTP boundary makes the tool server independently
#     testable and deployable
# ============================================================

import json
import httpx
from langchain_core.tools import tool

from utils.config import setup

config = setup()
MCP_SERVER_URL = config["mcp_server_url"]


# ── Tool: Search Clinical Guidelines ─────────────────────────

@tool
def search_guidelines(query: str) -> str:
    """
    Search the clinical guidelines corpus for evidence relevant to a
    clinical question. Returns the top matching guideline excerpts with
    their source titles and references.

    Use this tool for questions about screening recommendations,
    treatment thresholds, preventive care guidelines, and
    evidence-based clinical practice.

    Args:
        query: A natural language clinical question or topic.
    """
    try:
        response = httpx.post(
            f"{MCP_SERVER_URL}/tools/search_guidelines",
            json={"query": query, "k": 3},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        # Format results as readable text for the LLM
        formatted = []
        for i, result in enumerate(data["results"], 1):
            formatted.append(
                f"[Source {i}] {result['title']}\n"
                f"From: {result['source']}\n"
                f"Content: {result['content']}\n"
            )
        return "\n---\n".join(formatted) if formatted else "No relevant guidelines found."

    except httpx.ConnectError:
        return (
            "ERROR: Cannot connect to MCP tool server. "
            "Please ensure mcp_server.py is running on "
            f"{MCP_SERVER_URL}"
        )
    except Exception as e:
        return f"ERROR: Tool call failed — {str(e)}"


# ── Tool: Get Drug Information ────────────────────────────────

@tool
def get_drug_info(drug_name: str) -> str:
    """
    Retrieve drug label information for a named medication, including
    indications, contraindications, key drug interactions, and safety
    warnings. Information is sourced from FDA DailyMed drug labels.

    Use this tool for questions about specific medications: what they
    treat, when they should not be used, what they interact with,
    and what safety warnings apply.

    Args:
        drug_name: The name of the drug (generic or brand name).
    """
    try:
        response = httpx.post(
            f"{MCP_SERVER_URL}/tools/get_drug_info",
            json={"drug_name": drug_name},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        formatted = []
        for i, result in enumerate(data["results"], 1):
            formatted.append(
                f"[Source {i}] {result['title']}\n"
                f"From: {result['source']}\n"
                f"Content: {result['content']}\n"
            )
        return "\n---\n".join(formatted) if formatted else f"No drug information found for '{drug_name}'."

    except httpx.ConnectError:
        return (
            "ERROR: Cannot connect to MCP tool server. "
            "Please ensure mcp_server.py is running on "
            f"{MCP_SERVER_URL}"
        )
    except Exception as e:
        return f"ERROR: Tool call failed — {str(e)}"


# ── Tool Registry ─────────────────────────────────────────────

# Convenience list for binding to agents
GUIDELINES_TOOLS = [search_guidelines]
DRUG_INFO_TOOLS   = [get_drug_info]
ALL_TOOLS         = [search_guidelines, get_drug_info]
