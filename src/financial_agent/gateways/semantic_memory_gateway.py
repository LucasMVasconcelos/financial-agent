"""Semantic memory gateway — long-term, cross-conversation recall per customer.

Distinct from both the RAG knowledge base (`knowledge_base_gateway.py`,
shared static content about products/policies) and from
`ConversationRepository`'s rolling summary (bounded to *this* session's
older turns): this is memory that survives across separate conversations
entirely, and is retrieved by semantic similarity to the current message
rather than replayed in full.

Isolation is structural, not filter-based: each `user_id` gets its own
`InMemoryVectorStore`, so there is no shared index a query could ever leak
across — the same "identity never crosses a boundary the LLM controls"
principle used everywhere else in this project (see `agent/tools/base.py`),
applied at the storage layer instead of the tool layer.

Swapping this for a persistent backend (pgvector, Redis, a managed vector
DB) means implementing the same Protocol; nothing above this layer changes.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

DEFAULT_TOP_K = 3


class SemanticMemoryGateway(Protocol):
    async def remember(self, user_id: int, text: str) -> None:
        """Store a new long-term memory snippet for `user_id`."""
        ...

    async def recall(self, user_id: int, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[str]:
        """Return up to `top_k` memory snippets for `user_id` relevant to `query`."""
        ...


class InMemorySemanticMemoryGateway:
    """One in-memory vector store per user; created lazily on first write."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._stores: dict[int, InMemoryVectorStore] = {}

    async def remember(self, user_id: int, text: str) -> None:
        store = self._stores.setdefault(user_id, InMemoryVectorStore(self._embeddings))
        await store.aadd_texts([text])

    async def recall(self, user_id: int, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[str]:
        store = self._stores.get(user_id)
        if store is None:
            return []
        documents = await store.asimilarity_search(query, k=top_k)
        return [document.page_content for document in documents]
