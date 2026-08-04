from __future__ import annotations

import pytest

from financial_agent.domain.errors import ToolError as DomainToolError
from financial_agent.domain.errors import ToolErrorCode
from financial_agent.domain.models.knowledge import KnowledgeSnippet
from financial_agent.services.knowledge_base_service import KnowledgeBaseService


class _StubGateway:
    def __init__(self, *, snippets: list[KnowledgeSnippet] | None = None) -> None:
        self._snippets = snippets or []
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    async def search(self, query: str, *, top_k: int = 3) -> list[KnowledgeSnippet]:
        self.last_query = query
        self.last_top_k = top_k
        return self._snippets


class _FailingGateway:
    async def search(self, query: str, *, top_k: int = 3) -> list[KnowledgeSnippet]:
        raise DomainToolError(ToolErrorCode.UPSTREAM_ERROR, "vector store down")


class TestKnowledgeBaseService:
    async def test_happy_path_delegates_to_gateway(self) -> None:
        snippet = KnowledgeSnippet(
            title="Tesouro Selic", content="...", source="tesouro_selic", score=0.9
        )
        gateway = _StubGateway(snippets=[snippet])
        service = KnowledgeBaseService(gateway)

        results = await service.search("how does tesouro selic work")

        assert results == [snippet]
        assert gateway.last_query == "how does tesouro selic work"

    async def test_strips_whitespace_before_delegating(self) -> None:
        gateway = _StubGateway()
        service = KnowledgeBaseService(gateway)

        await service.search("  daily liquidity cd  ")

        assert gateway.last_query == "daily liquidity cd"

    @pytest.mark.parametrize("query", ["", "  ", "ab"])
    async def test_validation_error_for_too_short_query(self, query: str) -> None:
        gateway = _StubGateway()
        service = KnowledgeBaseService(gateway)

        with pytest.raises(DomainToolError) as exc_info:
            await service.search(query)

        assert exc_info.value.code == ToolErrorCode.VALIDATION_ERROR
        assert gateway.last_query is None  # never reached the gateway

    async def test_upstream_error_propagates_from_gateway(self) -> None:
        service = KnowledgeBaseService(_FailingGateway())

        with pytest.raises(DomainToolError) as exc_info:
            await service.search("credit portability")

        assert exc_info.value.code == ToolErrorCode.UPSTREAM_ERROR
