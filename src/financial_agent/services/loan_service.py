"""Loan origination service — orchestrates the LangGraph flow, the repository, and
the (best-effort) customer notification once a human decides.

Two entry points, matching the two places this gets invoked from:

  * `request_loan` — called from the `request_loan` Tool, inside a live
    Telegram request. Starts a new graph run; returns immediately whether
    that resolved (small amount, auto-approved) or paused
    (`PENDING_APPROVAL`, waiting on a human) — never blocks on a human.
  * `decide` — called from the admin approval endpoint
    (`api/routers/admin_loans.py`), completely outside any Telegram
    request. Resumes the checkpointed graph with the human's verdict, then
    notifies the customer proactively (the request that originated the
    loan is long gone by this point).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from financial_agent.agent.loan_graph import LoanGraphState
from financial_agent.domain.errors import NotFoundAppError, ValidationAppError
from financial_agent.domain.models.loan import LoanApplication, LoanStatus
from financial_agent.gateways.telegram_gateway import TelegramGateway
from financial_agent.observability.logging import get_logger
from financial_agent.repositories.loan_repository import LoanRepository

logger = get_logger(__name__)


class LoanService:
    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        loan_repository: LoanRepository,
        telegram_gateway: TelegramGateway,
    ) -> None:
        self._graph = graph
        self._loan_repository = loan_repository
        self._telegram_gateway = telegram_gateway

    async def request_loan(self, *, user_id: int, amount: float) -> LoanApplication:
        application_id = str(uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": application_id}}
        initial_state: LoanGraphState = {
            "application_id": application_id,
            "user_id": user_id,
            "amount": amount,
            "status": "new",
            "reason": "",
            "requires_human_approval": False,
            "human_decision": None,
            "decided_by": None,
        }
        result = await self._graph.ainvoke(initial_state, config)

        application = LoanApplication(
            application_id=application_id,
            user_id=user_id,
            amount=amount,
            status=LoanStatus(result["status"]),
            reason=result["reason"],
            requires_human_approval=result["requires_human_approval"],
            created_at=datetime.now(UTC),
        )
        await self._loan_repository.save(application)
        return application

    async def decide(
        self, *, application_id: str, approved: bool, decided_by: str
    ) -> LoanApplication:
        """Resume a paused application with a human verdict; notify the customer."""
        existing = await self._loan_repository.get_by_id(application_id)
        if existing is None:
            raise NotFoundAppError(f"Loan application '{application_id}' not found.")
        if existing.status is not LoanStatus.PENDING_APPROVAL:
            raise ValidationAppError(
                f"Application '{application_id}' is '{existing.status.value}', "
                "not pending approval."
            )

        config: RunnableConfig = {"configurable": {"thread_id": application_id}}
        await self._graph.aupdate_state(
            config,
            {"human_decision": "approved" if approved else "rejected", "decided_by": decided_by},
        )
        result = await self._graph.ainvoke(None, config)

        updated = existing.model_copy(
            update={
                "status": LoanStatus(result["status"]),
                "reason": result["reason"],
                "decided_by": decided_by,
                "decided_at": datetime.now(UTC),
            }
        )
        await self._loan_repository.save(updated)
        await self._notify_customer(updated)
        return updated

    async def list_by_status(self, status: LoanStatus) -> list[LoanApplication]:
        return await self._loan_repository.list_by_status(status)

    async def _notify_customer(self, application: LoanApplication) -> None:
        """Best-effort: the decision is already persisted even if the Telegram
        message fails to send — the customer isn't left without a decision on
        record, just without an immediate notification of it.
        """
        try:
            text = (
                f"Atualização sobre seu pedido de empréstimo de "
                f"R$ {application.amount:,.2f}: {application.reason}"
            )
            # In a private Telegram chat with the bot, chat_id == user_id.
            await self._telegram_gateway.send_message(chat_id=application.user_id, text=text)
        except Exception as exc:
            logger.warning(
                "loan_notification_failed",
                application_id=application.application_id,
                user_id=application.user_id,
                error=str(exc),
            )
