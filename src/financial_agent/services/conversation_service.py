"""Conversation history service — assembles the agent's memory for a turn.

Orchestrates all three memory kinds described in
`domain/models/conversation.py`:

  1. Loads the raw window + running summary from `ConversationRepository`.
  2. Overlays semantically-recalled snippets from *past* conversations
     (`SemanticMemoryService`), keyed by the current message — this is what
     makes `get_history` take `current_message` instead of just `user_id`.
  3. On `record_turn`, once the raw window grows past `_MAX_RAW_MESSAGES`,
     compacts the oldest `_COMPACT_CHUNK` messages into the running summary
     (via `ConversationSummarizer`) and files that summary away as a new
     long-term memory — a compacted chunk of conversation is exactly the
     kind of durable fact worth recalling in a future conversation.

Compaction is best-effort: it runs a real LLM call, and a transient failure
there must never block message delivery (the turn has already been decided
and is about to be sent to the customer). See the `except Exception` in
`_maybe_compact` — this is the one place in the service layer that
intentionally swallows an unstructured error rather than translating it
into a `ToolError`, because this call site isn't behind a Tool boundary.
"""

from __future__ import annotations

from financial_agent.agent.conversation_summarizer import ConversationSummarizer
from financial_agent.domain.models.conversation import (
    ConversationHistory,
    ConversationMessage,
    MessageRole,
)
from financial_agent.observability.logging import get_logger
from financial_agent.repositories.conversation_repository import (
    ConversationRepository,
    now_utc,
)
from financial_agent.services.semantic_memory_service import SemanticMemoryService

logger = get_logger(__name__)

_MAX_RAW_MESSAGES = 20
_COMPACT_CHUNK = 10


class ConversationService:
    def __init__(
        self,
        conversation_repository: ConversationRepository,
        summarizer: ConversationSummarizer,
        semantic_memory_service: SemanticMemoryService,
    ) -> None:
        self._conversation_repository = conversation_repository
        self._summarizer = summarizer
        self._semantic_memory_service = semantic_memory_service

    async def get_history(self, user_id: int, *, current_message: str) -> ConversationHistory:
        """Assemble this turn's memory: raw window + summary + long-term recall."""
        history = await self._conversation_repository.get_history(user_id)
        long_term_memories = await self._semantic_memory_service.recall(user_id, current_message)
        return history.model_copy(update={"long_term_memories": long_term_memories})

    async def record_turn(self, user_id: int, *, role: MessageRole, content: str) -> None:
        message = ConversationMessage(role=role, content=content, timestamp=now_utc())
        await self._conversation_repository.append_message(user_id, message)
        await self._maybe_compact(user_id)

    async def _maybe_compact(self, user_id: int) -> None:
        history = await self._conversation_repository.get_history(user_id)
        if len(history.messages) <= _MAX_RAW_MESSAGES:
            return

        to_summarize = history.messages[:_COMPACT_CHUNK]
        keep_last = len(history.messages) - _COMPACT_CHUNK
        try:
            new_summary = await self._summarizer.summarize(
                existing_summary=history.summary, new_turns=to_summarize
            )
            await self._conversation_repository.compact(
                user_id, keep_last=keep_last, summary=new_summary
            )
        except Exception as exc:
            logger.warning("conversation_compaction_failed", user_id=user_id, error=str(exc))
            return

        await self._semantic_memory_service.remember(user_id, new_summary)
