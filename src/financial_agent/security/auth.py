"""Authentication/authorization for non-Telegram (internal/service) callers.

Endpoints reached directly by operators or other internal services (not by
the Telegram webhook, which uses `telegram_auth` instead) require a static
API key sent via the `X-Service-Api-Key` header. This is intentionally
simple (no OAuth/JWT) because the only "service" caller in this project is
an operator hitting admin/health endpoints; swap for JWT/OAuth2 if this
grows real service-to-service traffic.
"""

from __future__ import annotations

import hmac

from fastapi import Header

from financial_agent.config import get_settings
from financial_agent.domain.errors import UnauthorizedAppError

SERVICE_API_KEY_HEADER = "X-Service-Api-Key"


def require_service_api_key(
    x_service_api_key: str | None = Header(default=None, alias=SERVICE_API_KEY_HEADER),
) -> None:
    """FastAPI dependency enforcing the service API key on protected routes."""
    settings = get_settings()
    if not x_service_api_key or not hmac.compare_digest(
        x_service_api_key, settings.service_api_key
    ):
        raise UnauthorizedAppError("Missing or invalid service API key.")
