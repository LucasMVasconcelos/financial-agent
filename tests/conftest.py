"""Shared pytest fixtures.

Async fixtures build a real `AppState` (real in-memory repositories, real
mock NBA gateway, real in-memory vector store) but never touch the network:
no test relies on an actual OpenAI or Telegram API call — the knowledge
base is embedded with a deterministic fake embeddings model instead of
`OpenAIEmbeddings`. Tests that reach the webhook monkeypatch the
agent-invocation and Telegram-sending call sites directly (see
`tests/api/test_telegram_webhook.py`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from financial_agent.api.app_state import AppState, build_app_state, shutdown_app_state
from financial_agent.config import Settings, get_settings
from financial_agent.main import app


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("SERVICE_API_KEY", "test-service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("APP_ENV", "local")
    # Tests must stay network-free regardless of what a developer's local
    # .env happens to set — without this, USE_REDIS=true in .env (e.g. left
    # on after a manual docker-compose/Redis Stack validation) makes every
    # test try to resolve the "redis" hostname and fail outside Docker.
    monkeypatch.setenv("USE_REDIS", "false")
    get_settings.cache_clear()

    # The knowledge base is embedded once at startup (build_app_state) — swap
    # in a deterministic, network-free embeddings model so no test ever
    # depends on a real OpenAI call to build the vector store.
    monkeypatch.setattr(
        "financial_agent.api.app_state.build_embeddings",
        lambda _settings: DeterministicFakeEmbedding(size=32),
    )


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest_asyncio.fixture
async def app_state(settings: Settings) -> AsyncIterator[AppState]:
    state = await build_app_state(settings)
    try:
        yield state
    finally:
        await shutdown_app_state(state)


@pytest_asyncio.fixture
async def client(app_state: AppState) -> AsyncIterator[AsyncClient]:
    app.state.container = app_state
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
