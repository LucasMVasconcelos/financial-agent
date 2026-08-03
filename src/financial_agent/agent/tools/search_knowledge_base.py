"""SearchKnowledgeBaseTool — Retrieval-Augmented Generation over product/policy articles.

Unlike the other three Tools (which fetch structured records for the
authenticated customer), this one answers open-ended "how does X work"
questions by semantically searching a small corpus of bank articles
(`gateways/knowledge_base_gateway.py`) and returning the most relevant
snippets for the LLM to ground its answer in — the classic RAG pattern.

`query` is free text and genuinely LLM-authored (there is no identity or
customer data in it), which is exactly the kind of input that's safe to let
the model control: it only narrows *what the tool searches for*, never
*whose data it reads*.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from financial_agent.agent.tools.base import run_tool
from financial_agent.services.knowledge_base_service import KnowledgeBaseService

SEARCH_KNOWLEDGE_BASE_TOOL_NAME = "search_knowledge_base"


class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(
        min_length=3,
        max_length=300,
        description=(
            "The customer's question or topic, in natural language, e.g. "
            "'how does credit portability work'. Do not include the "
            "customer's name or any identity information."
        ),
    )


class KnowledgeSnippetItem(BaseModel):
    title: str
    content: str
    source: str
    score: float


class SearchKnowledgeBaseOutput(BaseModel):
    results: list[KnowledgeSnippetItem]


def build_search_knowledge_base_tool(
    *, user_id: int, knowledge_base_service: KnowledgeBaseService
) -> StructuredTool:
    """`user_id` is bound only for logging/tracing correlation — no per-user filtering applies."""

    async def _handler(query: str) -> str:
        async def _call() -> SearchKnowledgeBaseOutput:
            snippets = await knowledge_base_service.search(query)
            return SearchKnowledgeBaseOutput(
                results=[
                    KnowledgeSnippetItem(
                        title=s.title, content=s.content, source=s.source, score=s.score
                    )
                    for s in snippets
                ]
            )

        return await run_tool(
            tool_name=SEARCH_KNOWLEDGE_BASE_TOOL_NAME, user_id=user_id, handler=_call
        )

    return StructuredTool.from_function(
        name=SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
        description=(
            "Searches the bank's knowledge base (product rules, policies, how-to articles) "
            "and returns the most relevant passages. Use this to answer 'how does X work' "
            "or 'what is X' questions — e.g. about CDs, Tesouro Selic, insurance, credit "
            "portability, early installment payoff — instead of answering from memory. Do not "
            "use this for the customer's personalized recommendation; use "
            "get_next_best_action for that."
        ),
        args_schema=SearchKnowledgeBaseInput,
        coroutine=_handler,
    )
