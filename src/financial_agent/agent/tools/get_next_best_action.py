"""GetNextBestActionTool — the core NBA recommendation tool.

Single responsibility: given the *authenticated* user, ask the NBA model
gateway for the next best financial action. Takes no LLM-controlled input
at all (there is nothing safe/meaningful for the LLM to choose here besides
"call this for the current user"), which is the strictest form of
identity-required dispatch.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from financial_agent.agent.tools.base import run_tool
from financial_agent.services.nba_service import NBAService

GET_NEXT_BEST_ACTION_TOOL_NAME = "get_next_best_action"


class GetNextBestActionInput(BaseModel):
    """No fields: the tool always acts on the authenticated caller."""


class GetNextBestActionOutput(BaseModel):
    action: str
    confidence: float
    reason: str


def build_get_next_best_action_tool(*, user_id: int, nba_service: NBAService) -> StructuredTool:
    """Bind `user_id` into the tool closure — never exposed to the LLM."""

    async def _handler(**_: object) -> str:
        async def _call() -> GetNextBestActionOutput:
            candidate = await nba_service.get_recommendation(user_id)
            return GetNextBestActionOutput(
                action=candidate.action.value,
                confidence=candidate.confidence,
                reason=candidate.reason,
            )

        return await run_tool(
            tool_name=GET_NEXT_BEST_ACTION_TOOL_NAME, user_id=user_id, handler=_call
        )

    return StructuredTool.from_function(
        name=GET_NEXT_BEST_ACTION_TOOL_NAME,
        description=(
            "Returns the single next-best financial action recommended for the "
            "current authenticated customer, with a confidence score (0-1) and a "
            "short factual reason. Always call this before recommending any "
            "financial action — never invent a recommendation yourself. "
            "Takes no arguments."
        ),
        args_schema=GetNextBestActionInput,
        coroutine=_handler,
    )
