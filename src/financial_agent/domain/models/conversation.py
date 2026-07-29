"""Conversation history domain models, used to give the agent short-term memory."""

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

    def as_transcript(self, *, max_messages: int = 20) -> list[ConversationMessage]:
        """Last N messages, oldest first — bounds prompt size."""
        return self.messages[-max_messages:]
