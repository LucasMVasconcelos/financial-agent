"""Conversation memory domain models.

Three distinct kinds of "memory" are modeled here, deliberately kept
separate because each has a different lifecycle:

  * `messages` — the raw, most-recent turns (short-term / working memory).
    Bounded by `ConversationService`'s compaction policy, not by this model.
  * `summary` — a running compression of everything older than the raw
    window, produced by `agent/conversation_summarizer.py` and persisted by
    the repository alongside the raw messages.
  * `long_term_memories` — snippets semantically recalled from *past*
    conversations (beyond this session's window entirely), via
    `services/semantic_memory_service.py`. Populated by the service layer,
    not stored on the repository — see `ConversationService.get_history`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str
    timestamp: datetime


class ConversationHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int
    messages: list[ConversationMessage] = Field(default_factory=list)
    summary: str | None = Field(
        default=None, description="Running summary of turns older than the raw window."
    )
    long_term_memories: list[str] = Field(
        default_factory=list,
        description="Snippets recalled from past conversations, relevant to the current turn.",
    )

    def as_transcript(self, *, max_messages: int = 20) -> list[ConversationMessage]:
        """Last N messages, oldest first — bounds prompt size."""
        return self.messages[-max_messages:]
