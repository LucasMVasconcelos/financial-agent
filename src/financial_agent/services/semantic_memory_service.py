"""Long-term semantic memory service.

Thin wrapper over `SemanticMemoryGateway` with one important property both
methods share: they are **best-effort**. Long-term recall is a context
enhancement, not a requirement for the agent to function — a failure here
(embeddings backend down, transient network error) must never break message
delivery or turn persistence. Compare with `services/customer_service.py`,
where a failure *does* propagate (the agent genuinely cannot proceed
without knowing who the customer is): the two failure philosophies differ
on purpose, matched to how essential each dependency actually is.
"""

from __future__ import annotations

from financial_agent.gateways.semantic_memory_gateway import DEFAULT_TOP_K, SemanticMemoryGateway
from financial_agent.observability.logging import get_logger

logger = get_logger(__name__)


class SemanticMemoryService:
    def __init__(self, semantic_memory_gateway: SemanticMemoryGateway) -> None:
        self._semantic_memory_gateway = semantic_memory_gateway

    async def remember(self, user_id: int, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            await self._semantic_memory_gateway.remember(user_id, text)
        except Exception as exc:
            logger.warning("semantic_memory_remember_failed", user_id=user_id, error=str(exc))

    async def recall(self, user_id: int, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[str]:
        query = query.strip()
        if not query:
            return []
        try:
            return await self._semantic_memory_gateway.recall(user_id, query, top_k=top_k)
        except Exception as exc:
            logger.warning("semantic_memory_recall_failed", user_id=user_id, error=str(exc))
            return []
