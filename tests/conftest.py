"""Shared pytest fixtures.

Async fixtures build a real `AppState` (real in-memory repositories, real
mock NBA gateway) but never touch the network: no test relies on an actual
OpenAI or Telegram API call. Tests that reach the webhook monkeypatch the
agent-invocation and Telegram-sending call sites directly (see
`tests/api/test_telegram_webhook.py`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

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
    get_settings.cache_clear()


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
