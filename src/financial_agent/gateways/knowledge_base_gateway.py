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
        "title": "Daily-Liquidity CD",
        "content": (
            "A Daily-Liquidity CD (CDB Liquidez Diária) is a bank deposit certificate that "
            "can be redeemed at any time, with no loss of yield proportional to the period "
            "invested. It typically pays a percentage of the CDI rate and is guaranteed by "
            "the FGC (Brazil's deposit insurance fund) up to the legal limit of "
            "R$ 250,000.00 per taxpayer ID (CPF) and institution."
        ),
    },
    {
        "source": "tesouro_selic",
        "title": "Tesouro Selic",
        "content": (
            "Tesouro Selic is a floating-rate government bond that tracks the Selic policy "
            "rate. It is recommended for an emergency fund thanks to its low risk and good "
            "liquidity, with redemption within one business day. Regressive income tax "
            "applies to the yield."
        ),
    },
    {
        "source": "seguro_vida",
        "title": "Life Insurance",
        "content": (
            "Life Insurance guarantees a payout to the named beneficiaries in the event of "
            "the insured's death or disability. The monthly premium depends on age, the "
            "chosen insured amount, and additional coverage, such as critical illness."
        ),
    },
    {
        "source": "seguro_residencial",
        "title": "Homeowners Insurance",
        "content": (
            "Homeowners Insurance covers damage to the property and belongings inside it "
            "from fire, theft, flooding, and other events covered by the policy. It can "
            "include 24-hour assistance for plumbers, electricians, and locksmiths."
        ),
    },
    {
        "source": "cartao_black_limite",
        "title": "Black Card Limit Increase",
        "content": (
            "A Black Card limit increase is evaluated automatically based on usage history, "
            "on-time payments, and the relationship with the bank. The customer can also "
            "request a manual limit review through the app."
        ),
    },
    {
        "source": "portabilidade_credito",
        "title": "Credit Portability",
        "content": (
            "Credit portability lets a customer transfer a loan or financing from another "
            "institution to this bank, usually seeking a lower interest rate. There is no "
            "cost for the customer to request portability, per the Central Bank of Brazil "
            "(Bacen)'s regulation."
        ),
    },
    {
        "source": "reserva_emergencia",
        "title": "Emergency Fund",
        "content": (
            "An emergency fund is an amount set aside in highly liquid investments, such as "
            "Tesouro Selic or a daily-liquidity CD, recommended to cover 3 to 6 months of "
            "essential expenses in case of the unexpected."
        ),
    },
    {
        "source": "antecipacao_parcelas",
        "title": "Early Installment Payoff",
        "content": (
            "Early installment payoff lets a customer settle part or all of a loan ahead of "
            "schedule, with a proportional discount on future interest as required by "
            "current law. It can be done directly through the app, with no need for human "
            "assistance."
        ),
    },
)


class KnowledgeBaseGateway(Protocol):
    async def search(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[KnowledgeSnippet]:
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

    async def search(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[KnowledgeSnippet]:
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
