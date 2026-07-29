"""Telegram webhook authentication.

Telegram does not sign webhook payloads by default. The recommended (and
only reliable) way to verify that a request actually came from Telegram's
servers is the `secret_token` mechanism: when registering the webhook via
`setWebhook(url=..., secret_token=SECRET)`, Telegram echoes that same value
back on every subsequent call in the `X-Telegram-Bot-Api-Secret-Token`
header. We compare it with a constant-time comparison to avoid timing
side-channels, and reject anything that doesn't match — before the payload
is even parsed.
"""

from __future__ import annotations

import hmac

TELEGRAM_SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"


def is_valid_telegram_secret(*, received: str | None, expected: str) -> bool:
    """Constant-time comparison of the webhook secret token header."""
    if not received or not expected:
        return False
    return hmac.compare_digest(received, expected)
