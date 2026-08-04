"""Unit tests for `SemanticMemoryService` — in particular its best-effort contract:
a failing gateway must never raise past this service.
"""

from __future__ import annotations

from financial_agent.services.semantic_memory_service import SemanticMemoryService


class _StubGateway:
    def __init__(self) -> None:
        self.remembered: list[tuple[int, str]] = []

    async def remember(self, user_id: int, text: str) -> None:
        self.remembered.append((user_id, text))

    async def recall(self, user_id: int, query: str, *, top_k: int = 3) -> list[str]:
        return [f"memory-for-{query}"]


class _FailingGateway:
    async def remember(self, user_id: int, text: str) -> None:
        raise RuntimeError("vector store unavailable")

    async def recall(self, user_id: int, query: str, *, top_k: int = 3) -> list[str]:
        raise RuntimeError("vector store unavailable")


class TestSemanticMemoryService:
    async def test_remember_delegates_to_gateway(self) -> None:
        gateway = _StubGateway()
        service = SemanticMemoryService(gateway)

        await service.remember(123, "customer asked about CDs")

        assert gateway.remembered == [(123, "customer asked about CDs")]

    async def test_remember_skips_blank_text(self) -> None:
        gateway = _StubGateway()
        service = SemanticMemoryService(gateway)

        await service.remember(123, "   ")

        assert gateway.remembered == []

    async def test_recall_delegates_to_gateway(self) -> None:
        service = SemanticMemoryService(_StubGateway())

        results = await service.recall(123, "portability")

        assert results == ["memory-for-portability"]

    async def test_recall_skips_blank_query(self) -> None:
        service = SemanticMemoryService(_StubGateway())

        assert await service.recall(123, "  ") == []

    async def test_remember_swallows_gateway_failures(self) -> None:
        service = SemanticMemoryService(_FailingGateway())

        await service.remember(123, "something")  # must not raise

    async def test_recall_swallows_gateway_failures_and_returns_empty(self) -> None:
        service = SemanticMemoryService(_FailingGateway())

        results = await service.recall(123, "something")

        assert results == []
