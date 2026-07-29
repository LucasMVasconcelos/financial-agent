"""Correlation/request-id middleware + basic HTTP access logging.

Accepts an inbound `X-Correlation-Id` (so an upstream gateway/load balancer
can propagate one end-to-end); generates one otherwise. Always mints a fresh
`X-Request-Id` per HTTP call. Both are echoed back on the response and
merged into every structlog line for the duration of the request via
`observability.context` contextvars.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from financial_agent.observability.context import (
    get_correlation_id,
    get_request_id,
    set_request_context,
)
from financial_agent.observability.logging import get_logger

logger = get_logger(__name__)

CORRELATION_ID_HEADER = "X-Correlation-Id"
REQUEST_ID_HEADER = "X-Request-Id"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        set_request_context(correlation_id=request.headers.get(CORRELATION_ID_HEADER))
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers[CORRELATION_ID_HEADER] = get_correlation_id()
        response.headers[REQUEST_ID_HEADER] = get_request_id()

        logger.info(
            "http_request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response
