"""Conversation history service — the agent's short-term memory."""

from __future__ import annotations

from financial_agent.domain.models.conversation import (
    ConversationHistory,
    ConversationMessage,
    MessageRole,
)
from financial_agent.repositories.conversation_repository import (
    ConversationRepository,
    now_utc,
)


class ConversationService:
    def __init__(self, conversation_repository: ConversationRepository) -> None:
        self._conversation_repository = conversation_repository

    async def get_history(self, user_id: int) -> ConversationHistory:
        return await self._conversation_repository.get_history(user_id)

    async def record_turn(self, user_id: int, *, role: MessageRole, content: str) -> None:
        message = ConversationMessage(role=role, content=content, timestamp=now_utc())
        await self._conversation_repository.append_message(user_id, message)
