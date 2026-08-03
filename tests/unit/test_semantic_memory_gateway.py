"""Unit tests for `InMemorySemanticMemoryGateway`.

Uses `DeterministicFakeEmbedding` (network-free) — same pattern as
`test_knowledge_base_gateway.py`. The property under test is isolation:
each user gets their own vector store, so one user's memories can never
leak into another user's recall results.
"""

from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding

from financial_agent.gateways.semantic_memory_gateway import InMemorySemanticMemoryGateway

USER_A = 123
USER_B = 456


def _build_gateway() -> InMemorySemanticMemoryGateway:
    return InMemorySemanticMemoryGateway(DeterministicFakeEmbedding(size=32))


class TestInMemorySemanticMemoryGateway:
    async def test_recall_before_any_remember_returns_empty(self) -> None:
        gateway = _build_gateway()

        results = await gateway.recall(USER_A, "anything")

        assert results == []

    async def test_remember_then_recall_returns_the_stored_text(self) -> None:
        gateway = _build_gateway()

        await gateway.remember(USER_A, "Customer asked about credit portability.")

        results = await gateway.recall(USER_A, "credit portability", top_k=3)

        assert results == ["Customer asked about credit portability."]

    async def test_memories_are_isolated_per_user(self) -> None:
        gateway = _build_gateway()

        await gateway.remember(USER_A, "Fact about user A.")
        await gateway.remember(USER_B, "Fact about user B.")

        results_a = await gateway.recall(USER_A, "fact", top_k=5)
        results_b = await gateway.recall(USER_B, "fact", top_k=5)

        assert results_a == ["Fact about user A."]
        assert results_b == ["Fact about user B."]

    async def test_top_k_is_respected(self) -> None:
        gateway = _build_gateway()

        for i in range(5):
            await gateway.remember(USER_A, f"Memory number {i}.")

        results = await gateway.recall(USER_A, "memory", top_k=2)

        assert len(results) == 2
