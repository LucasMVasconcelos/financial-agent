"""Knowledge base domain models — the RAG retrieval result shape."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeSnippet(BaseModel):
    """One retrieved passage from the knowledge base, with its similarity score."""

    model_config = ConfigDict(frozen=True)

    title: str
    content: str
    source: str = Field(description="Stable article id, e.g. 'cdb_liquidez_diaria'.")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score (higher is more relevant).")
