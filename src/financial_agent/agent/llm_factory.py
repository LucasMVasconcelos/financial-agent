"""Chat Model construction and LangSmith tracing bootstrap.

Centralizing model construction here means swapping providers (OpenAI ->
Azure OpenAI, Anthropic, a local vLLM endpoint, ...) touches one file. All
LangChain `Runnable`s in this project (the tool-calling agent, the filler
chain) are built from the same `ChatModel` instance type returned here.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from financial_agent.config import Settings


def build_chat_model(settings: Settings, *, temperature: float | None = None) -> ChatOpenAI:
    """Construct the LangChain Chat Model used by every agent Runnable."""
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature if temperature is None else temperature,
        api_key=settings.openai_api_key,
        timeout=20.0,
        max_retries=2,
    )


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Construct the Embeddings model backing `search_knowledge_base` (RAG)."""
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model, openai_api_key=settings.openai_api_key
    )


def configure_langsmith_tracing(settings: Settings) -> None:
    """Propagate LangSmith settings into env vars, which LangChain reads directly.

    A no-op (tracing stays off) unless `LANGCHAIN_TRACING_V2=true` and an API
    key are configured — safe to call unconditionally at startup.
    """
    if not settings.langchain_tracing_v2:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
