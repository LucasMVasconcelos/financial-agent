"""Unit tests for LoanService: origination, human decision (resume), and the
best-effort customer notification.
"""

from __future__ import annotations

import pytest

from financial_agent.agent.loan_graph import build_loan_graph
from financial_agent.domain.errors import NotFoundAppError, ValidationAppError
from financial_agent.domain.models.loan import LoanStatus
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.repositories.loan_repository import InMemoryLoanRepository
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.loan_service import LoanService

KNOWN_USER_ID = 123
_THRESHOLD = 50_000.0


class _StubTelegramGateway:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[tuple[int, str]] = []
        self._fail = fail

    async def send_message(self, *, chat_id: int, text: str) -> None:
        if self._fail:
            raise RuntimeError("telegram is down")
        self.sent.append((chat_id, text))


def _build_service(*, telegram_fails: bool = False) -> tuple[LoanService, _StubTelegramGateway]:
    customer_repo = InMemoryCustomerRepository()
    customer_service = CustomerService(customer_repo)
    graph = build_loan_graph(customer_service=customer_service, approval_threshold=_THRESHOLD)
    telegram = _StubTelegramGateway(fail=telegram_fails)
    service = LoanService(
        graph=graph, loan_repository=InMemoryLoanRepository(), telegram_gateway=telegram
    )
    return service, telegram


class TestLoanServiceRequestLoan:
    async def test_small_amount_auto_approves(self) -> None:
        service, _ = _build_service()

        application = await service.request_loan(user_id=KNOWN_USER_ID, amount=1_000.0)

        assert application.status is LoanStatus.APPROVED
        assert application.requires_human_approval is False

    async def test_large_amount_is_pending_approval(self) -> None:
        service, _ = _build_service()

        application = await service.request_loan(user_id=KNOWN_USER_ID, amount=80_000.0)

        assert application.status is LoanStatus.PENDING_APPROVAL
        assert application.requires_human_approval is True

    async def test_application_is_persisted_and_retrievable_via_list(self) -> None:
        service, _ = _build_service()
        application = await service.request_loan(user_id=KNOWN_USER_ID, amount=80_000.0)

        pending = await service.list_by_status(LoanStatus.PENDING_APPROVAL)

        assert [a.application_id for a in pending] == [application.application_id]


class TestLoanServiceDecide:
    async def test_decide_unknown_application_raises_not_found(self) -> None:
        service, _ = _build_service()

        with pytest.raises(NotFoundAppError):
            await service.decide(application_id="does-not-exist", approved=True, decided_by="ana")

    async def test_decide_already_decided_application_raises_validation_error(self) -> None:
        service, _ = _build_service()
        application = await service.request_loan(user_id=KNOWN_USER_ID, amount=1_000.0)

        with pytest.raises(ValidationAppError):
            await service.decide(
                application_id=application.application_id, approved=True, decided_by="ana"
            )

    async def test_approve_disburses_and_notifies_customer(self) -> None:
        service, telegram = _build_service()
        application = await service.request_loan(user_id=KNOWN_USER_ID, amount=80_000.0)

        decided = await service.decide(
            application_id=application.application_id, approved=True, decided_by="ana.analista"
        )

        assert decided.status is LoanStatus.DISBURSED
        assert decided.decided_by == "ana.analista"
        assert decided.decided_at is not None
        assert len(telegram.sent) == 1
        assert telegram.sent[0][0] == KNOWN_USER_ID

    async def test_reject_updates_status_and_notifies_customer(self) -> None:
        service, telegram = _build_service()
        application = await service.request_loan(user_id=KNOWN_USER_ID, amount=80_000.0)

        decided = await service.decide(
            application_id=application.application_id, approved=False, decided_by="ana.analista"
        )

        assert decided.status is LoanStatus.REJECTED
        assert len(telegram.sent) == 1

    async def test_decided_application_is_persisted(self) -> None:
        service, _ = _build_service()
        application = await service.request_loan(user_id=KNOWN_USER_ID, amount=80_000.0)
        await service.decide(
            application_id=application.application_id, approved=True, decided_by="ana"
        )

        approved = await service.list_by_status(LoanStatus.DISBURSED)
        assert [a.application_id for a in approved] == [application.application_id]

    async def test_notification_failure_does_not_break_decide(self) -> None:
        service, _ = _build_service(telegram_fails=True)
        application = await service.request_loan(user_id=KNOWN_USER_ID, amount=80_000.0)

        decided = await service.decide(
            application_id=application.application_id, approved=True, decided_by="ana"
        )  # must not raise

        assert decided.status is LoanStatus.DISBURSED
