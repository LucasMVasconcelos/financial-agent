"""Knowledge base (RAG) service — validates input and delegates to the vector store gateway."""

from __future__ import annotations

from financial_agent.domain.errors import ToolError, ToolErrorCode
from financial_agent.domain.models.knowledge import KnowledgeSnippet
from financial_agent.gateways.knowledge_base_gateway import DEFAULT_TOP_K, KnowledgeBaseGateway

_MIN_QUERY_LENGTH = 3


class KnowledgeBaseService:
    def __init__(self, knowledge_base_gateway: KnowledgeBaseGateway) -> None:
        self._knowledge_base_gateway = knowledge_base_gateway

    async def search(
        self, query: str, *, top_k: int = DEFAULT_TOP_K
    ) -> list[KnowledgeSnippet]:
        normalized_query = query.strip()
        if len(normalized_query) < _MIN_QUERY_LENGTH:
            raise ToolError(
                ToolErrorCode.VALIDATION_ERROR,
                "Query must have at least 3 characters.",
                details={"query": query},
            )
        return await self._knowledge_base_gateway.search(normalized_query, top_k=top_k)
