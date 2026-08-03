"""Admin loan approval endpoints — the human half of the human-in-the-loop flow.

Not reachable by Telegram or by the LLM in any way; these are for whoever
reviews pending loan applications (an ops/credit analyst). Protected by the
same `X-Service-Api-Key` mechanism used for any other internal/operator
route (`security/auth.py`) — no new auth concept introduced for this.

Deciding an application resumes the checkpointed LangGraph run
(`agent/loan_graph.py` via `LoanService.decide`) and proactively notifies
the customer over Telegram — the request that originally asked for the
loan is long gone by the time a human acts on it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from financial_agent.api.deps import get_loan_service
from financial_agent.domain.models.loan import LoanStatus
from financial_agent.security.auth import require_service_api_key
from financial_agent.services.loan_service import LoanService

router = APIRouter(
    prefix="/admin/loans",
    tags=["admin"],
    dependencies=[Depends(require_service_api_key)],
)


class LoanDecisionRequest(BaseModel):
    approved: bool
    decided_by: str


class LoanApplicationResponse(BaseModel):
    application_id: str
    user_id: int
    amount: float
    status: LoanStatus
    reason: str
    requires_human_approval: bool
    decided_by: str | None
    created_at: str
    decided_at: str | None


@router.get("")
async def list_loan_applications(
    status: LoanStatus = Query(default=LoanStatus.PENDING_APPROVAL),
    loan_service: LoanService = Depends(get_loan_service),
) -> list[LoanApplicationResponse]:
    applications = await loan_service.list_by_status(status)
    return [
        LoanApplicationResponse(
            application_id=a.application_id,
            user_id=a.user_id,
            amount=a.amount,
            status=a.status,
            reason=a.reason,
            requires_human_approval=a.requires_human_approval,
            decided_by=a.decided_by,
            created_at=a.created_at.isoformat(),
            decided_at=a.decided_at.isoformat() if a.decided_at else None,
        )
        for a in applications
    ]


@router.post("/{application_id}/decide")
async def decide_loan_application(
    application_id: str,
    body: LoanDecisionRequest,
    loan_service: LoanService = Depends(get_loan_service),
) -> LoanApplicationResponse:
    application = await loan_service.decide(
        application_id=application_id, approved=body.approved, decided_by=body.decided_by
    )
    return LoanApplicationResponse(
        application_id=application.application_id,
        user_id=application.user_id,
        amount=application.amount,
        status=application.status,
        reason=application.reason,
        requires_human_approval=application.requires_human_approval,
        decided_by=application.decided_by,
        created_at=application.created_at.isoformat(),
        decided_at=application.decided_at.isoformat() if application.decided_at else None,
    )
