"""Request-scoped correlation/request IDs, propagated via contextvars.

Using `contextvars` (rather than thread-locals) matters because FastAPI runs
handlers as asyncio tasks: a plain thread-local would leak state across
concurrent requests. Values set here are automatically merged into every
structlog log line (see `observability.logging.configure_logging`) without
having to thread a `correlation_id` parameter through every function call.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def new_id() -> str:
    return uuid.uuid4().hex


def set_request_context(*, correlation_id: str | None, request_id: str | None = None) -> None:
    _correlation_id.set(correlation_id or new_id())
    _request_id.set(request_id or new_id())


def get_correlation_id() -> str:
    return _correlation_id.get()


def get_request_id() -> str:
    return _request_id.get()
