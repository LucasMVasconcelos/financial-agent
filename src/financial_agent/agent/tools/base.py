"""Shared Tool contract: envelope, error translation, identity-bound dispatch.

Every Tool in this project follows the same shape:

  1. **Narrow input schema** — a Pydantic model with only the fields the LLM
     is actually allowed to choose. `user_id` (or any other identity field)
     is *never* a field on these schemas.
  2. **Identity-required dispatch** — the concrete `StructuredTool` instance
     is built per-request by a factory function that closes over the
     authenticated `user_id` (see `agent/tools/get_next_best_action.py` etc,
     and `agent/agent_executor.py` where the factories are invoked). The LLM
     can never inject or override whose data it reads.
  3. **Narrow, strongly-typed output schema**, wrapped in `ToolEnvelope` so
     both success and structured-error paths share one serialization shape.
  4. **No exception ever escapes to the agent loop** — `run_tool` catches
     `ToolError` (expected, structured) and any other exception (translated
     to `UNKNOWN_ERROR`), logs it, and always returns a `ToolEnvelope`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pydantic import BaseModel

from financial_agent.domain.errors import ToolError as DomainToolError
from financial_agent.domain.errors import ToolErrorCode
from financial_agent.observability.logging import get_logger
from financial_agent.observability.tracing import traced_span

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ToolErrorPayload(BaseModel):
    code: ToolErrorCode
    message: str
    details: dict[str, object] = {}


class ToolEnvelope(BaseModel, Generic[T]):
    """Uniform success/error envelope returned by every Tool, as JSON."""

    success: bool
    data: T | None = None
    error: ToolErrorPayload | None = None

    @classmethod
    def ok(cls, data: T) -> ToolEnvelope[T]:
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, error: DomainToolError) -> ToolEnvelope[T]:
        return cls(
            success=False,
            data=None,
            error=ToolErrorPayload(code=error.code, message=error.message, details=error.details),
        )


async def run_tool(
    *,
    tool_name: str,
    user_id: int,
    handler: Callable[[], Awaitable[T]],
) -> str:
    """Execute `handler`, always returning a serialized `ToolEnvelope`.

    This is the single place that (a) times the call, (b) logs it with
    correlation context, and (c) guarantees the agent loop never sees a raw
    exception — satisfying "never let an exception interrupt the
    conversation".
    """
    with traced_span(f"tool.{tool_name}", user_id=user_id):
        try:
            result = await handler()
            envelope: ToolEnvelope[T] = ToolEnvelope.ok(result)
        except DomainToolError as exc:
            logger.warning(
                "tool_structured_error", tool=tool_name, user_id=user_id, code=exc.code
            )
            envelope = ToolEnvelope.fail(exc)
        except Exception as exc:
            logger.error(
                "tool_unexpected_error", tool=tool_name, user_id=user_id, error=str(exc)
            )
            envelope = ToolEnvelope.fail(
                DomainToolError(ToolErrorCode.UNKNOWN_ERROR, "Unexpected internal error.")
            )
        return envelope.model_dump_json()
