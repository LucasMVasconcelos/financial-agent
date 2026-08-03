"""Loan origination domain models.

This is the project's one *mutating* capability — every other Tool is a
read-only lookup. `LoanApplication` tracks the lifecycle of a request end to
end: `PENDING_APPROVAL` is a genuine waiting state (a human hasn't decided
yet), not a transient in-request status — see `agent/loan_graph.py` for how
that wait is actually implemented (a checkpointed LangGraph run, not a
blocked request).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LoanStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISBURSED = "disbursed"


class LoanApplication(BaseModel):
    model_config = ConfigDict(frozen=True)

    application_id: str
    user_id: int
    amount: float = Field(gt=0)
    status: LoanStatus
    reason: str = Field(description="Human-readable explanation of the current status.")
    requires_human_approval: bool
    decided_by: str | None = Field(
        default=None, description="Identifier of the analyst who approved/rejected, if any."
    )
    created_at: datetime
    decided_at: datetime | None = None
