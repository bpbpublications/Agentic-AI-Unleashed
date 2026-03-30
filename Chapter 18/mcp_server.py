# mcp_server.py
# ============================================================
# MCP-Compatible Tool Server (FastAPI Implementation)
# Agentic AI Unleashed — Appendix Demo
#
# This FastAPI server simulates an MCP (Model Context Protocol)
# tool server. It exposes two clinical tools over HTTP that the
# LangGraph agents call via LangChain tool wrappers.
#
# What is MCP?
#   The Model Context Protocol defines a standard interface
#   for LLMs to discover and invoke external tools. A real MCP
#   server uses the official MCP SDK and communicates over
#   stdio or SSE transport. Here we implement the same *concept*
#   — tools exposed over a protocol boundary, callable by any
#   agent — using FastAPI for simplicity and transparency.
#
# In production you would replace this with:
#   from mcp.server.fastmcp import FastMCP
#   mcp = FastMCP("clinical-tools")
#   @mcp.tool() ...
#
# Run this server before starting main.py:
#   python mcp_server.py
# ============================================================

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.pipeline import retrieve, build_index
from utils.config import setup

# ── Initialise ──────────────────────────────────────────────
config = setup()
app = FastAPI(
    title="Clinical Research MCP Tool Server",
    description="MCP-compatible tool server for the Clinical Research Assistant Agent demo.",
    version="1.0.0",
)

# Build the FAISS index at server startup
@app.on_event("startup")
async def startup_event():
    build_index()
    print("[MCP Server] Ready — tools: search_guidelines, get_drug_info")


# ── Request / Response Models ────────────────────────────────

class SearchGuidelinesRequest(BaseModel):
    query: str = Field(description="Clinical question or topic to search guidelines for.")
    k: int = Field(default=3, description="Number of results to return.")

class SearchGuidelinesResponse(BaseModel):
    results: list[dict] = Field(description="Top-k relevant clinical guideline chunks.")
    query:   str         = Field(description="The original query.")

class GetDrugInfoRequest(BaseModel):
    drug_name: str = Field(description="Name of the drug to look up.")

class GetDrugInfoResponse(BaseModel):
    results: list[dict] = Field(description="Retrieved drug information chunks.")
    drug_name: str       = Field(description="The queried drug name.")


# ── Tool Endpoints ───────────────────────────────────────────

@app.post(
    "/tools/search_guidelines",
    response_model=SearchGuidelinesResponse,
    summary="Search clinical guidelines",
    description=(
        "Performs semantic search over the clinical guidelines corpus using FAISS. "
        "Returns the top-k most relevant evidence chunks for the given query."
    ),
)
async def search_guidelines(request: SearchGuidelinesRequest) -> SearchGuidelinesResponse:
    """
    MCP Tool: search_guidelines
    ---------------------------
    Retrieves relevant clinical guideline chunks for a given query.
    Used by the Guidelines Agent to ground its responses in evidence.
    """
    results = retrieve(query=request.query, k=request.k)
    return SearchGuidelinesResponse(results=results, query=request.query)


@app.post(
    "/tools/get_drug_info",
    response_model=GetDrugInfoResponse,
    summary="Get drug information",
    description=(
        "Retrieves drug information from the FDA DailyMed-sourced corpus. "
        "Returns relevant chunks covering indications, contraindications, "
        "interactions, and safety notes."
    ),
)
async def get_drug_info(request: GetDrugInfoRequest) -> GetDrugInfoResponse:
    """
    MCP Tool: get_drug_info
    -----------------------
    Retrieves drug label information for a named drug.
    Used by the Drug Information Agent to answer pharmacology queries.
    The query is enriched with 'drug information' to bias retrieval
    toward drug-specific chunks rather than clinical guidelines.
    """
    enriched_query = f"{request.drug_name} drug information indications interactions"
    results = retrieve(query=enriched_query, k=3)
    return GetDrugInfoResponse(results=results, drug_name=request.drug_name)


# ── Health Check ─────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "server": "Clinical Research MCP Tool Server"}


@app.get("/tools", summary="List available tools")
async def list_tools():
    """MCP-style tool manifest — lists available tools and their descriptions."""
    return {
        "tools": [
            {
                "name":        "search_guidelines",
                "endpoint":    "/tools/search_guidelines",
                "description": "Semantic search over clinical guidelines corpus (USPSTF, ADA, ACC/AHA).",
                "input_schema": SearchGuidelinesRequest.model_json_schema(),
            },
            {
                "name":        "get_drug_info",
                "endpoint":    "/tools/get_drug_info",
                "description": "Retrieve drug label information from FDA DailyMed-sourced corpus.",
                "input_schema": GetDrugInfoRequest.model_json_schema(),
            },
        ]
    }


# ── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(config.get("mcp_server_port", 8001))
    print(f"[MCP Server] Starting on http://localhost:{port}")
    print(f"[MCP Server] Tool manifest: http://localhost:{port}/tools")
    print(f"[MCP Server] API docs:      http://localhost:{port}/docs")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=port, reload=False)
