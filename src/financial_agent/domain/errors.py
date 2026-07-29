"""Structured error contract shared by every Tool and API boundary.

Tools must never let a raw exception bubble up into the LangChain agent loop
(an uncaught exception there aborts the conversation turn). Instead, tool
handlers catch domain/infra exceptions and translate them into a
`ToolError`, which the Tool wrapper (see `financial_agent.agent.tools.base`)
serializes into a `ToolEnvelope` that is handed back to the LLM as a normal
observation. The LLM can then read the error `code` and decide how to react
(retry, apologize, ask a clarifying question, etc).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ToolErrorCode(StrEnum):
    """Closed set of error codes tools are allowed to raise.

    Kept intentionally small and provider-agnostic so the agent's reasoning
    (and its prompt instructions) can special-case each one without needing
    to know about upstream implementation details (HTTP status codes,
    database driver exceptions, etc).
    """

    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ToolError(Exception):
    """Raised by services/gateways and caught at the Tool boundary.

    `details` must only ever hold non-sensitive, already-sanitized data:
    it may be surfaced to the LLM and, transitively, to the end user.
    """

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class AppError(Exception):
    """Base class for errors raised at the API layer (outside tool calls)."""

    def __init__(self, code: ToolErrorCode, message: str, *, http_status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class UnauthorizedAppError(AppError):
    def __init__(self, message: str = "Unauthorized") -> None:
        super().__init__(ToolErrorCode.UNAUTHORIZED, message, http_status=401)


class ValidationAppError(AppError):
    def __init__(self, message: str = "Invalid request payload") -> None:
        super().__init__(ToolErrorCode.VALIDATION_ERROR, message, http_status=422)


class RateLimitedAppError(AppError):
    def __init__(self, message: str = "Too many requests") -> None:
        super().__init__(ToolErrorCode.RATE_LIMITED, message, http_status=429)
