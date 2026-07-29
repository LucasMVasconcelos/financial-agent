"""Thin async HTTP client over the Telegram Bot API.

Only the two calls this project needs are exposed: sending a chat message
and sending a "typing..." action (used to keep the user engaged while the
NBA tool call and the LLM turn are in flight — see
`financial_agent.agent.filler_agent`). All outbound identity is the bot's
own token; there is no user-supplied data in the auth path.
"""

from __future__ import annotations

import httpx

from financial_agent.domain.errors import ToolError, ToolErrorCode
from financial_agent.observability.logging import get_logger

logger = get_logger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramGateway:
    def __init__(self, *, bot_token: str, http_client: httpx.AsyncClient | None = None) -> None:
        self._base_url = f"{TELEGRAM_API_BASE}/bot{bot_token}"
        self._client = http_client or httpx.AsyncClient(timeout=10.0)

    async def send_message(self, *, chat_id: int, text: str) -> None:
        try:
            response = await self._client.post(
                f"{self._base_url}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("telegram_send_message_failed", chat_id=chat_id, error=str(exc))
            raise ToolError(
                ToolErrorCode.UPSTREAM_ERROR, "Failed to deliver message via Telegram."
            ) from exc

    async def send_typing_action(self, *, chat_id: int) -> None:
        try:
            response = await self._client.post(
                f"{self._base_url}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # Non-critical UX nicety: log and continue rather than failing the turn.
            logger.warning("telegram_typing_action_failed", chat_id=chat_id, error=str(exc))

    async def aclose(self) -> None:
        await self._client.aclose()
