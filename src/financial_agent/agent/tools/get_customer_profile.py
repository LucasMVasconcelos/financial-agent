"""GetCustomerProfileTool — grounds the agent in who the customer is.

Also takes no LLM-controlled input; the profile returned is always the
authenticated caller's own. Output is deliberately narrower than the
internal `CustomerProfile` domain model (no `user_id`, no `created_at`) —
the LLM only gets what it needs to reason about a recommendation.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from financial_agent.agent.tools.base import run_tool
from financial_agent.services.customer_service import CustomerService

GET_CUSTOMER_PROFILE_TOOL_NAME = "get_customer_profile"


class GetCustomerProfileInput(BaseModel):
    """No fields: the tool always acts on the authenticated caller."""


class GetCustomerProfileOutput(BaseModel):
    full_name: str
    segment: str
    risk_profile: str
    account_balance: float
    owned_products: list[str]


def build_get_customer_profile_tool(
    *, user_id: int, customer_service: CustomerService
) -> StructuredTool:
    async def _handler(**_: object) -> str:
        async def _call() -> GetCustomerProfileOutput:
            profile = await customer_service.get_profile(user_id)
            return GetCustomerProfileOutput(
                full_name=profile.full_name,
                segment=profile.segment,
                risk_profile=profile.risk_profile.value,
                account_balance=profile.account_balance,
                owned_products=[p.name for p in profile.products],
            )

        return await run_tool(
            tool_name=GET_CUSTOMER_PROFILE_TOOL_NAME, user_id=user_id, handler=_call
        )

    return StructuredTool.from_function(
        name=GET_CUSTOMER_PROFILE_TOOL_NAME,
        description=(
            "Returns the current authenticated customer's profile: name, segment, "
            "risk profile, checking-account balance, and products already owned. "
            "Use this to personalize explanations and avoid recommending products "
            "the customer already has. Takes no arguments."
        ),
        args_schema=GetCustomerProfileInput,
        coroutine=_handler,
    )
