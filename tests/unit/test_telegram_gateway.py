from __future__ import annotations

import httpx
import pytest
import respx

from financial_agent.domain.errors import ToolError, ToolErrorCode
from financial_agent.gateways.telegram_gateway import TelegramGateway


@pytest.fixture
def gateway() -> TelegramGateway:
    return TelegramGateway(bot_token="test-token")


class TestTelegramGateway:
    @respx.mock
    async def test_send_message_happy_path(self, gateway: TelegramGateway) -> None:
        route = respx.post(
            "https://api.telegram.org/bottest-token/sendMessage"
        ).mock(return_value=httpx.Response(200, json={"ok": True}))

        await gateway.send_message(chat_id=42, text="hello")

        assert route.called

    @respx.mock
    async def test_send_message_upstream_error_on_failure(
        self, gateway: TelegramGateway
    ) -> None:
        respx.post("https://api.telegram.org/bottest-token/sendMessage").mock(
            return_value=httpx.Response(500, json={"ok": False})
        )

        with pytest.raises(ToolError) as exc_info:
            await gateway.send_message(chat_id=42, text="hello")

        assert exc_info.value.code == ToolErrorCode.UPSTREAM_ERROR

    @respx.mock
    async def test_typing_action_failure_does_not_raise(self, gateway: TelegramGateway) -> None:
        respx.post("https://api.telegram.org/bottest-token/sendChatAction").mock(
            return_value=httpx.Response(500)
        )

        await gateway.send_typing_action(chat_id=42)  # must not raise
