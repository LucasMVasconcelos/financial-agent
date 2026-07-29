"""Knowledge base gateway — RAG retrieval over a small corpus of product/policy articles.

This is the project's one genuinely retrieval-augmented-generation component
(the other three Tools do structured data lookups, not semantic search). The
seam is the same shape as `NBAModelGateway`: a narrow `Protocol`
(`async def search(query, top_k) -> list[KnowledgeSnippet]`), with
`InMemoryKnowledgeBaseGateway` as the concrete implementation.

`InMemoryVectorStore` (from `langchain_core`) holds the embedded corpus for
this demo — fine for a handful of short articles, but it re-embeds nothing
across restarts and doesn't scale or persist. Swapping in a real vector
database (pgvector, Pinecone, OpenSearch, ...) means writing one new class
against this same Protocol; nothing above the gateway layer changes.

The corpus is embedded once at startup (`InMemoryKnowledgeBaseGateway.build`,
called from `api/app_state.py`) rather than per-request, since re-embedding
static content on every search would be wasteful and slow.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from financial_agent.domain.errors import ToolError as DomainToolError
from financial_agent.domain.errors import ToolErrorCode
from financial_agent.domain.models.knowledge import KnowledgeSnippet
from financial_agent.observability.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_K = 3

_ARTICLES: tuple[dict[str, str], ...] = (
    {
        "source": "cdb_liquidez_diaria",
        "title": "CDB Liquidez Diária",
        "content": (
            "O CDB Liquidez Diária é um Certificado de Depósito Bancário que permite resgate "
            "a qualquer momento, sem perda de rentabilidade proporcional ao período investido. "
            "Costuma render um percentual do CDI e é garantido pelo FGC até o limite legal de "
            "R$ 250.000,00 por CPF e instituição."
        ),
    },
    {
        "source": "tesouro_selic",
        "title": "Tesouro Selic",
        "content": (
            "O Tesouro Selic é um título público pós-fixado que acompanha a taxa Selic. É "
            "indicado para reserva de emergência por ter baixo risco e boa liquidez, com "
            "resgate em até um dia útil. Incide Imposto de Renda regressivo sobre o rendimento."
        ),
    },
    {
        "source": "seguro_vida",
        "title": "Seguro de Vida",
        "content": (
            "O Seguro de Vida garante uma indenização aos beneficiários indicados em caso de "
            "morte ou invalidez do segurado. O valor do prêmio mensal depende da idade, do "
            "capital segurado escolhido e de coberturas adicionais, como doenças graves."
        ),
    },
    {
        "source": "seguro_residencial",
        "title": "Seguro Residencial",
        "content": (
            "O Seguro Residencial cobre danos ao imóvel e a bens dentro dele por incêndio, "
            "roubo, alagamento e outros eventos previstos em apólice. Pode incluir assistência "
            "24h para encanador, eletricista e chaveiro."
        ),
    },
    {
        "source": "cartao_black_limite",
        "title": "Aumento de Limite do Cartão Black",
        "content": (
            "O aumento de limite do Cartão Black é avaliado automaticamente com base no "
            "histórico de uso, pagamento em dia e relacionamento com o banco. O cliente pode "
            "também solicitar uma revisão manual de limite pelo aplicativo."
        ),
    },
    {
        "source": "portabilidade_credito",
        "title": "Portabilidade de Crédito",
        "content": (
            "A portabilidade de crédito permite transferir um empréstimo ou financiamento de "
            "outra instituição para o banco, geralmente buscando uma taxa de juros menor. Não "
            "há custo para o cliente solicitar a portabilidade, conforme regulação do Bacen."
        ),
    },
    {
        "source": "reserva_emergencia",
        "title": "Reserva de Emergência",
        "content": (
            "A reserva de emergência é uma quantia guardada em aplicações de alta liquidez, "
            "como Tesouro Selic ou CDB de liquidez diária, recomendada para cobrir de 3 a 6 "
            "meses de despesas essenciais em caso de imprevistos."
        ),
    },
    {
        "source": "antecipacao_parcelas",
        "title": "Antecipação de Parcelas",
        "content": (
            "A antecipação de parcelas permite quitar parte ou o total de um empréstimo antes "
            "do prazo, com desconto proporcional de juros futuros conforme legislação vigente. "
            "Pode ser feita diretamente pelo aplicativo, sem necessidade de atendimento humano."
        ),
    },
)


class KnowledgeBaseGateway(Protocol):
    async def search(
        self, query: str, *, top_k: int = DEFAULT_TOP_K
    ) -> list[KnowledgeSnippet]:
        """Return the `top_k` most relevant snippets for `query`.

        Raises:
            ToolError: with code UPSTREAM_ERROR if the vector store/embedding
                backend fails.
        """
        ...


class InMemoryKnowledgeBaseGateway:
    """Vector-store-backed RAG gateway over the static corpus in `_ARTICLES`."""

    def __init__(self, vector_store: InMemoryVectorStore) -> None:
        self._vector_store = vector_store

    @classmethod
    async def build(cls, embeddings: Embeddings) -> InMemoryKnowledgeBaseGateway:
        """Embed the corpus once and return a ready-to-query gateway."""
        documents = [
            Document(
                page_content=article["content"],
                metadata={"title": article["title"], "source": article["source"]},
            )
            for article in _ARTICLES
        ]
        vector_store = await InMemoryVectorStore.afrom_documents(documents, embeddings)
        return cls(vector_store)

    async def search(
        self, query: str, *, top_k: int = DEFAULT_TOP_K
    ) -> list[KnowledgeSnippet]:
        try:
            results = await self._vector_store.asimilarity_search_with_score(query, k=top_k)
        except Exception as exc:
            logger.error("knowledge_base_search_failed", error=str(exc))
            raise DomainToolError(
                ToolErrorCode.UPSTREAM_ERROR, "Knowledge base is unavailable."
            ) from exc

        return [
            KnowledgeSnippet(
                title=str(document.metadata.get("title", "")),
                content=document.page_content,
                source=str(document.metadata.get("source", "")),
                score=round(min(max(score, 0.0), 1.0), 4),
            )
            for document, score in results
        ]
