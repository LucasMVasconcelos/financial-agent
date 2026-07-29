"""API tests for the Telegram webhook endpoint.

Covers: authentication (missing/invalid secret header), payload validation,
non-text updates, the happy path, and per-user rate limiting. The LLM-backed
agent and the outbound Telegram HTTP calls are monkeypatched at the router's
import sites — these tests exercise the webhook's *contract*, not LangChain
or the live Telegram API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from financial_agent.agent.output_parser import FillerReply
from financial_agent.api.app_state import AppState
from financial_agent.security.telegram_auth import TELEGRAM_SECRET_HEADER

WEBHOOK_URL = "/webhook/telegram"
VALID_SECRET = "test-webhook-secret"


def _update_payload(*, text: str = "Olá, quero uma recomendação", user_id: int = 123) -> dict:
    return {
        "update_id": 1,
        "message": {
            "message_id": 10,
            "from": {"id": user_id, "is_bot": False, "first_name": "Ana"},
            "chat": {"id": user_id},
            "text": text,
        },
    }


@pytest.fixture(autouse=True)
def _stub_agent_and_telegram(
    monkeypatch: pytest.MonkeyPatch, app_state: AppState
) -> None:
    """Replace the LLM-backed agent turn and outbound Telegram calls with stubs."""
    monkeypatch.setattr(
        "financial_agent.api.routers.telegram_webhook.build_agent_executor",
        lambda **_kwargs: object(),
    )

    async def _fake_run_agent_turn(**_kwargs: object) -> str:
        return "Recomendamos investir em CDB."

    monkeypatch.setattr(
        "financial_agent.api.routers.telegram_webhook.run_agent_turn", _fake_run_agent_turn
    )

    app_state.filler_agent.generate = AsyncMock(  # type: ignore[method-assign]
        return_value=FillerReply(message="Já te respondo!")
    )
    app_state.telegram_gateway.send_message = AsyncMock()  # type: ignore[method-assign]
    app_state.telegram_gateway.send_typing_action = AsyncMock()  # type: ignore[method-assign]


class TestTelegramWebhookAuth:
    async def test_missing_secret_header_returns_401(self, client: AsyncClient) -> None:
        response = await client.post(WEBHOOK_URL, json=_update_payload())

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    async def test_invalid_secret_header_returns_401(self, client: AsyncClient) -> None:
        response = await client.post(
            WEBHOOK_URL,
            json=_update_payload(),
            headers={TELEGRAM_SECRET_HEADER: "wrong-secret"},
        )

        assert response.status_code == 401


class TestTelegramWebhookPayload:
    async def test_invalid_payload_returns_422(self, client: AsyncClient) -> None:
        response = await client.post(
            WEBHOOK_URL,
            json={"update_id": "not-an-int"},
            headers={TELEGRAM_SECRET_HEADER: VALID_SECRET},
        )

        assert response.status_code == 422

    async def test_non_text_update_is_acknowledged_and_ignored(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        payload = {"update_id": 2}  # no "message" at all (e.g. a non-message update)

        response = await client.post(
            WEBHOOK_URL, json=payload, headers={TELEGRAM_SECRET_HEADER: VALID_SECRET}
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        app_state.telegram_gateway.send_message.assert_not_called()  # type: ignore[attr-defined]


class TestTelegramWebhookHappyPath:
    async def test_valid_payload_sends_reply_and_persists_history(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        response = await client.post(
            WEBHOOK_URL,
            json=_update_payload(user_id=123),
            headers={TELEGRAM_SECRET_HEADER: VALID_SECRET},
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}

        # Two sends are expected by design: the filler acknowledgement (sent
        # concurrently while the agent works) and the final agent reply.
        send_message_mock = app_state.telegram_gateway.send_message  # type: ignore[attr-defined]
        assert send_message_mock.await_count == 2
        sent_texts = [call.kwargs["text"] for call in send_message_mock.await_args_list]
        assert "Recomendamos investir em CDB." in sent_texts

        history = await app_state.conversation_service.get_history(123)
        assert [m.content for m in history.messages] == [
            "Olá, quero uma recomendação",
            "Recomendamos investir em CDB.",
        ]


class TestTelegramWebhookRateLimit:
    async def test_exceeding_quota_returns_429(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        app_state.rate_limiter._max_requests = 1  # type: ignore[attr-defined]

        first = await client.post(
            WEBHOOK_URL,
            json=_update_payload(user_id=456),
            headers={TELEGRAM_SECRET_HEADER: VALID_SECRET},
        )
        second = await client.post(
            WEBHOOK_URL,
            json=_update_payload(user_id=456),
            headers={TELEGRAM_SECRET_HEADER: VALID_SECRET},
        )

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "RATE_LIMITED"
