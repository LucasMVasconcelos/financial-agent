"""Unit tests for the loan LangGraph flow (agent/loan_graph.py).

Exercises the checkpointer-based pause/resume directly: a small amount
auto-approves in one `ainvoke`; a large amount pauses at
`pending_approval`, and injecting a human decision via `aupdate_state` +
re-invoking with `None` resumes and finalizes — the same pattern
`LoanService` uses in production.
"""

from __future__ import annotations

from financial_agent.agent.loan_graph import build_loan_graph
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.services.customer_service import CustomerService

KNOWN_USER_ID = 123
_THRESHOLD = 50_000.0


def _build_graph(threshold: float = _THRESHOLD):
    repo = InMemoryCustomerRepository()
    return build_loan_graph(
        customer_service=CustomerService(repo), approval_threshold=threshold
    )


def _initial_state(application_id: str, amount: float) -> dict:
    return {
        "application_id": application_id,
        "user_id": KNOWN_USER_ID,
        "amount": amount,
        "status": "new",
        "reason": "",
        "requires_human_approval": False,
        "human_decision": None,
        "decided_by": None,
    }


class TestLoanGraphAutoApprove:
    async def test_amount_below_threshold_auto_approves_in_one_pass(self) -> None:
        graph = _build_graph()
        config = {"configurable": {"thread_id": "loan-small"}}

        result = await graph.ainvoke(_initial_state("loan-small", 1_000.0), config)

        assert result["status"] == "approved"
        assert result["requires_human_approval"] is False

    async def test_amount_exactly_at_threshold_auto_approves(self) -> None:
        graph = _build_graph()
        config = {"configurable": {"thread_id": "loan-exact"}}

        result = await graph.ainvoke(_initial_state("loan-exact", _THRESHOLD), config)

        assert result["status"] == "approved"


class TestLoanGraphHumanApproval:
    async def test_amount_above_threshold_pauses_pending_approval(self) -> None:
        graph = _build_graph()
        config = {"configurable": {"thread_id": "loan-big"}}

        result = await graph.ainvoke(_initial_state("loan-big", 80_000.0), config)

        assert result["status"] == "pending_approval"
        assert result["requires_human_approval"] is True

    async def test_resume_with_approval_disburses(self) -> None:
        graph = _build_graph()
        config = {"configurable": {"thread_id": "loan-approve"}}
        await graph.ainvoke(_initial_state("loan-approve", 80_000.0), config)

        await graph.aupdate_state(config, {"human_decision": "approved", "decided_by": "ana"})
        result = await graph.ainvoke(None, config)

        assert result["status"] == "disbursed"

    async def test_resume_with_rejection_rejects(self) -> None:
        graph = _build_graph()
        config = {"configurable": {"thread_id": "loan-reject"}}
        await graph.ainvoke(_initial_state("loan-reject", 80_000.0), config)

        await graph.aupdate_state(config, {"human_decision": "rejected", "decided_by": "ana"})
        result = await graph.ainvoke(None, config)

        assert result["status"] == "rejected"

    async def test_different_thread_ids_do_not_interfere(self) -> None:
        graph = _build_graph()
        cfg_a = {"configurable": {"thread_id": "loan-a"}}
        cfg_b = {"configurable": {"thread_id": "loan-b"}}

        await graph.ainvoke(_initial_state("loan-a", 80_000.0), cfg_a)
        await graph.ainvoke(_initial_state("loan-b", 1_000.0), cfg_b)

        await graph.aupdate_state(cfg_a, {"human_decision": "approved", "decided_by": "ana"})
        result_a = await graph.ainvoke(None, cfg_a)
        state_b = await graph.aget_state(cfg_b)

        assert result_a["status"] == "disbursed"
        assert state_b.values["status"] == "approved"
