# Clinical Research Assistant Agent
### Appendix Code — *Agentic AI Unleashed*
**Anand Oka & Jayaram Nanduri**

---

This repository contains the complete, runnable source code for the book's
appendix demonstration. It is intentionally designed for **clarity and learning**
rather than production hardening — every architectural decision is made to
illuminate a concept.

## What This Demo Illustrates

| Concept | Implementation | File |
|---|---|---|
| **MCP Tool Server** | FastAPI server exposing tools over HTTP | `mcp_server.py` |
| **RAG Pipeline** | FAISS in-memory vector search over clinical docs | `rag/pipeline.py` |
| **Multi-Agent (A2A pattern)** | LangGraph supervisor + specialist agents | `agents/graph.py` |
| **Memory** | LangGraph MemorySaver checkpointer | `agents/graph.py` |
| **Human-in-the-Loop** | LangGraph interrupt_before review node | `agents/graph.py` |
| **Structured Output** | Pydantic response models | `agents/models.py` |
| **Observability** | LangSmith automatic tracing | `.env` + `utils/config.py` |

## Repository Structure

```
clinical-research-agent/
│
├── mcp_server.py          # FastAPI MCP-compatible tool server (start this first)
│
├── rag/
│   ├── __init__.py
│   ├── corpus.py          # Curated open-source clinical document chunks
│   └── pipeline.py        # FAISS index builder and retrieval function
│
├── agents/
│   ├── __init__.py
│   ├── models.py          # Pydantic structured output models
│   ├── tools.py           # LangChain tools that call the MCP server
│   └── graph.py           # LangGraph multi-agent supervisor graph (core)
│
├── utils/
│   ├── __init__.py
│   └── config.py          # Environment loading and LangSmith setup
│
├── main.py                # Entry point — runs a sample multi-turn conversation
├── requirements.txt
├── .env.example           # Copy to .env and fill in your keys
└── README.md
```

## ⚠️ Environment Note — Read Before Running

This directory has its **own independent Python environment**, separate from the rest of the book's codebase. Do not attempt to run this agent from the book's root environment — the dependencies are managed independently using [uv](https://docs.astral.sh/uv).

Always `cd` into this directory first and run `uv sync` to create the local `.venv` before doing anything else:

```bash
cd appendix/
uv sync
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows
```

This ensures the correct versions of LangGraph, FAISS, FastAPI, and all other dependencies are isolated and consistent with what the appendix code expects.

## Quick Start

```bash
# 1. Clone and set up the environment (requires uv — https://docs.astral.sh/uv)
git clone https://github.com/your-repo/clinical-research-agent
cd clinical-research-agent
uv sync                          # creates .venv and installs all dependencies
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 2. Configure environment
cp .env.example .env
# Edit .env with your OpenAI and LangSmith keys

# 3. Start the MCP tool server (in a separate terminal)
python mcp_server.py

# 4. Run the scripted demo
python main.py

# 4b. Or run interactively (real HITL prompts, enter your own queries)
python main.py --interactive
```

## Extending to Production A2A

The inter-agent calls in `agents/graph.py` use LangGraph's internal
node-to-node routing. Comments throughout the file mark exactly where
these calls could be replaced with HTTP-based Agent-to-Agent (A2A)
protocol calls — as defined by Google's A2A specification — to enable
agents running in separate services or containers to collaborate.

---

*This code is provided as a learning companion to the book.
It is not intended for clinical use.*
