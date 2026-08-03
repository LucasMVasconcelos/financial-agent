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

        results = await gateway.recall(USER_A, "qualquer coisa")

        assert results == []

    async def test_remember_then_recall_returns_the_stored_text(self) -> None:
        gateway = _build_gateway()

        await gateway.remember(USER_A, "Cliente perguntou sobre portabilidade de crédito.")

        results = await gateway.recall(USER_A, "portabilidade de crédito", top_k=3)

        assert results == ["Cliente perguntou sobre portabilidade de crédito."]

    async def test_memories_are_isolated_per_user(self) -> None:
        gateway = _build_gateway()

        await gateway.remember(USER_A, "Fato sobre o usuário A.")
        await gateway.remember(USER_B, "Fato sobre o usuário B.")

        results_a = await gateway.recall(USER_A, "fato", top_k=5)
        results_b = await gateway.recall(USER_B, "fato", top_k=5)

        assert results_a == ["Fato sobre o usuário A."]
        assert results_b == ["Fato sobre o usuário B."]

    async def test_top_k_is_respected(self) -> None:
        gateway = _build_gateway()

        for i in range(5):
            await gateway.remember(USER_A, f"Memória número {i}.")

        results = await gateway.recall(USER_A, "memória", top_k=2)

        assert len(results) == 2
