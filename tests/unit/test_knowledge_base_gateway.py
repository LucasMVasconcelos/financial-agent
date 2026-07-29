"""Unit tests for `InMemoryKnowledgeBaseGateway`.

Uses `DeterministicFakeEmbedding` (network-free, seeded by text hash) so
these tests never call the real OpenAI embeddings API — they verify the
gateway's structural contract (result count, score bounds, field mapping),
not semantic relevance, which requires real embeddings and is out of scope
for a unit test.
"""

from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding

from financial_agent.gateways.knowledge_base_gateway import InMemoryKnowledgeBaseGateway

_KNOWN_SOURCES = {
    "cdb_liquidez_diaria",
    "tesouro_selic",
    "seguro_vida",
    "seguro_residencial",
    "cartao_black_limite",
    "portabilidade_credito",
    "reserva_emergencia",
    "antecipacao_parcelas",
}


async def _build_gateway() -> InMemoryKnowledgeBaseGateway:
    return await InMemoryKnowledgeBaseGateway.build(DeterministicFakeEmbedding(size=32))


class TestInMemoryKnowledgeBaseGateway:
    async def test_search_returns_up_to_top_k_snippets(self) -> None:
        gateway = await _build_gateway()

        results = await gateway.search("como funciona a portabilidade de credito", top_k=3)

        assert 1 <= len(results) <= 3
        for snippet in results:
            assert snippet.source in _KNOWN_SOURCES
            assert snippet.title
            assert snippet.content
            assert 0.0 <= snippet.score <= 1.0

    async def test_top_k_is_respected(self) -> None:
        gateway = await _build_gateway()

        results = await gateway.search("seguro", top_k=1)

        assert len(results) == 1
