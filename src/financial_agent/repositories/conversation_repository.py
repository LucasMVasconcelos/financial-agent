"""Conversation history repository.

Backs the agent's short-term memory. `InMemoryConversationRepository` is
process-local (lost on restart, not shared across replicas) — acceptable for
a demo; swap for a Redis- or Postgres-backed implementation for production
by implementing the same Protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from financial_agent.domain.models.conversation import ConversationHistory, ConversationMessage


class ConversationRepository(Protocol):
    async def get_history(self, user_id: int) -> ConversationHistory:
        """Return the stored history for `user_id` (empty if none yet)."""
        ...

    async def append_message(self, user_id: int, message: ConversationMessage) -> None:
        """Persist a new turn for `user_id`."""
        ...


class InMemoryConversationRepository:
    def __init__(self) -> None:
        self._histories: dict[int, list[ConversationMessage]] = {}

    async def get_history(self, user_id: int) -> ConversationHistory:
        return ConversationHistory(user_id=user_id, messages=self._histories.get(user_id, []))

    async def append_message(self, user_id: int, message: ConversationMessage) -> None:
        self._histories.setdefault(user_id, []).append(message)


def now_utc() -> datetime:
    return datetime.now(UTC)
