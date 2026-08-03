"""Conversation history repository.

Backs the agent's short-term (working) memory: raw recent turns plus the
running summary that stands in for everything older. `compact()` is how
`ConversationService` folds the oldest raw messages into that summary and
trims them from storage — see `services/conversation_service.py` for the
policy that decides *when* to call it; this repository only knows how to
store the result.

`InMemoryConversationRepository` is process-local (lost on restart, not
shared across replicas) — acceptable for a demo; swap for a Redis- or
Postgres-backed implementation for production by implementing the same
Protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from financial_agent.domain.models.conversation import ConversationHistory, ConversationMessage


class ConversationRepository(Protocol):
    async def get_history(self, user_id: int) -> ConversationHistory:
        """Return the stored raw messages + running summary for `user_id`."""
        ...

    async def append_message(self, user_id: int, message: ConversationMessage) -> None:
        """Persist a new turn for `user_id`."""
        ...

    async def compact(self, user_id: int, *, keep_last: int, summary: str) -> None:
        """Replace the running summary and trim raw messages to the last `keep_last`."""
        ...


class InMemoryConversationRepository:
    def __init__(self) -> None:
        self._messages: dict[int, list[ConversationMessage]] = {}
        self._summaries: dict[int, str] = {}

    async def get_history(self, user_id: int) -> ConversationHistory:
        return ConversationHistory(
            user_id=user_id,
            messages=self._messages.get(user_id, []),
            summary=self._summaries.get(user_id),
        )

    async def append_message(self, user_id: int, message: ConversationMessage) -> None:
        self._messages.setdefault(user_id, []).append(message)

    async def compact(self, user_id: int, *, keep_last: int, summary: str) -> None:
        existing = self._messages.get(user_id, [])
        self._messages[user_id] = existing[-keep_last:] if keep_last > 0 else []
        self._summaries[user_id] = summary


def now_utc() -> datetime:
    return datetime.now(UTC)
