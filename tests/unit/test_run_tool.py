"""Unit tests for the shared Tool wrapper (`agent/tools/base.run_tool`).

Covers the required error-code matrix (RATE_LIMITED, NOT_FOUND,
UPSTREAM_ERROR, VALIDATION_ERROR/"invalid input", plus the happy path and
the last-resort UNKNOWN_ERROR translation) — the same contract every
concrete Tool handler relies on.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from financial_agent.agent.tools.base import run_tool
from financial_agent.domain.errors import ToolError as DomainToolError
from financial_agent.domain.errors import ToolErrorCode


class _Payload(BaseModel):
    value: str


class TestRunTool:
    async def test_happy_path_returns_success_envelope(self) -> None:
        async def handler() -> _Payload:
            return _Payload(value="ok")

        raw = await run_tool(tool_name="dummy", user_id=1, handler=handler)
        envelope = json.loads(raw)

        assert envelope == {"success": True, "data": {"value": "ok"}, "error": None}

    @pytest.mark.parametrize(
        "code",
        [
            ToolErrorCode.RATE_LIMITED,
            ToolErrorCode.NOT_FOUND,
            ToolErrorCode.UPSTREAM_ERROR,
            ToolErrorCode.VALIDATION_ERROR,
            ToolErrorCode.UNAUTHORIZED,
        ],
    )
    async def test_structured_errors_are_serialized(self, code: ToolErrorCode) -> None:
        async def handler() -> _Payload:
            raise DomainToolError(code, "something went wrong")

        raw = await run_tool(tool_name="dummy", user_id=1, handler=handler)
        envelope = json.loads(raw)

        assert envelope["success"] is False
        assert envelope["data"] is None
        assert envelope["error"]["code"] == code.value

    async def test_unexpected_exception_becomes_unknown_error(self) -> None:
        async def handler() -> _Payload:
            raise RuntimeError("boom")

        raw = await run_tool(tool_name="dummy", user_id=1, handler=handler)
        envelope = json.loads(raw)

        assert envelope["success"] is False
        assert envelope["error"]["code"] == ToolErrorCode.UNKNOWN_ERROR.value
        # The raw exception message must never leak to the LLM/end user.
        assert "boom" not in raw
