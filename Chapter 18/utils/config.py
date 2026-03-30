# utils/config.py
# ============================================================
# Configuration loader and observability bootstrap.
# All LangSmith wiring happens here — just by loading the
# environment variables, every LangGraph node execution,
# tool call, and token count flows automatically into traces.
# ============================================================

import os
import warnings
from dotenv import load_dotenv

# Suppress cosmetic Pydantic serialization warnings surfaced by LangChain internals.
# PydanticSerializationUnexpectedValue fires when LangChain passes non-Pydantic objects
# (e.g. raw dicts) through Pydantic serializers — it does not affect correctness.
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)

def setup() -> dict:
    """
    Load environment variables and initialize LangSmith tracing.
    Returns a config dict for use across the application.
    
    LangSmith tracing requires no code changes beyond setting:
      LANGCHAIN_TRACING_V2=true
      LANGCHAIN_API_KEY=<your key>
    LangChain/LangGraph reads these automatically on import.
    """
    load_dotenv()

    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_model":   os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "mcp_server_url": os.getenv("MCP_SERVER_URL", "http://localhost:8001"),
        "langsmith_enabled": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "clinical-research-agent"),
    }

    if config["langsmith_enabled"]:
        print(f"[Observability] LangSmith tracing active → project: '{config['langsmith_project']}'")
        print(f"[Observability] View traces at: https://smith.langchain.com")
    else:
        print("[Observability] LangSmith tracing disabled. Set LANGCHAIN_TRACING_V2=true to enable.")

    return config
