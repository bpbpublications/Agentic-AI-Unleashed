# agents/models.py
# ============================================================
# Pydantic models for structured agent responses.
# Using typed outputs rather than raw strings forces agents
# to be explicit about what they know, what they retrieved,
# and how confident they are — a key production habit.
# ============================================================

from pydantic import BaseModel, Field
from typing import Literal


class ClinicalResponse(BaseModel):
    """Structured response returned by the Guidelines Agent."""

    recommendation: str = Field(
        description="The clinical recommendation or finding, in plain language."
    )
    evidence_sources: list[str] = Field(
        description="List of source document titles or identifiers used."
    )
    confidence_level: Literal["high", "moderate", "low"] = Field(
        description="Agent's confidence based on retrieved evidence quality."
    )
    requires_human_review: bool = Field(
        description="Whether this response should be reviewed before delivery."
    )


class DrugInfoResponse(BaseModel):
    """Structured response returned by the Drug Information Agent."""

    drug_name: str = Field(description="Canonical drug name.")
    indication: str = Field(description="Primary approved indication(s).")
    key_interactions: list[str] = Field(
        description="Notable drug interactions relevant to the query."
    )
    safety_note: str = Field(
        description="Important safety consideration or black-box warning if applicable."
    )
    evidence_sources: list[str] = Field(
        description="Source document titles or identifiers."
    )


class SupervisorDecision(BaseModel):
    """Structured routing decision made by the Supervisor Agent."""

    route_to: Literal["guidelines_agent", "drug_agent", "respond_directly"] = Field(
        description="Which specialist agent should handle this query, or direct response."
    )
    reasoning: str = Field(
        description="Brief explanation of why this routing decision was made."
    )
