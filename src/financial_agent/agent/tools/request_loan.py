"""RequestLoanTool — the project's one mutating action.

Unlike the other four Tools, this one has a real, consequential side
effect: it originates a loan application, which may auto-approve or pause
for human review (see `agent/loan_graph.py`). It still follows the same
contract as every other Tool — narrow input, identity-bound dispatch,
`ToolEnvelope`/structured errors via `run_tool` — but its output is
explicitly a *status*, never a guarantee: `requires_human_approval=True`
means the customer has not been approved yet, and the agent must say so
plainly rather than implying the loan is done.

`amount` is free for the LLM to set — same reasoning as `query` on
`search_knowledge_base`: it carries no identity and doesn't reach into
another customer's data, it only parameterizes *this* customer's own
request.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from financial_agent.agent.tools.base import run_tool
from financial_agent.services.loan_service import LoanService

REQUEST_LOAN_TOOL_NAME = "request_loan"


class RequestLoanInput(BaseModel):
    amount: float = Field(
        gt=0,
        le=10_000_000,
        description="The loan amount requested by the customer, in BRL.",
    )


class RequestLoanOutput(BaseModel):
    application_id: str
    status: str
    reason: str
    requires_human_approval: bool


def build_request_loan_tool(*, user_id: int, loan_service: LoanService) -> StructuredTool:
    async def _handler(amount: float) -> str:
        async def _call() -> RequestLoanOutput:
            application = await loan_service.request_loan(user_id=user_id, amount=amount)
            return RequestLoanOutput(
                application_id=application.application_id,
                status=application.status.value,
                reason=application.reason,
                requires_human_approval=application.requires_human_approval,
            )

        return await run_tool(tool_name=REQUEST_LOAN_TOOL_NAME, user_id=user_id, handler=_call)

    return StructuredTool.from_function(
        name=REQUEST_LOAN_TOOL_NAME,
        description=(
            "Originates a loan request for the current authenticated customer. "
            "Consider calling get_next_best_action first to ground the offer in a "
            "real recommendation before proposing a loan. Returns a status, never a "
            "guarantee: if requires_human_approval is true, the loan is NOT yet "
            "approved — it is pending human review (this happens for amounts above "
            "the bank's auto-approval threshold) — tell the customer clearly that "
            "it is under analysis, not approved. Never state a loan is approved or "
            "disbursed unless the returned status says so explicitly."
        ),
        args_schema=RequestLoanInput,
        coroutine=_handler,
    )
