"""Loan origination flow — the project's one stateful, checkpointed, Plan-and-Execute-style flow.

Every other capability in this agent is a single read-only Tool call,
handled fine by the ReAct loop in `agent/main_graph.py`: decide, act,
observe, repeat. Loan origination is different in kind, not just degree — it is the
project's only *mutating* action, its outcome can require an indeterminate
wait for a human decision, and that wait must survive well past the
lifetime of the Telegram request that triggered it. A ReAct tool call can't
express "pause here, maybe for hours, then resume exactly where you left
off" — that is precisely what LangGraph's checkpointer is for, and why this
one flow is built as an explicit graph instead of another `StructuredTool`
handler.

The graph is intentionally plan-shaped rather than reactive: `assess` runs
once, `route_by_amount` decides the whole path upfront (auto-approve vs.
human review) based on `Settings.loan_human_approval_threshold`, and nothing
re-negotiates that decision later — the only thing a second invocation can
do is supply the human's verdict and let `finalize` run. Compare with
`agent/main_graph.py`'s docstring on why the ReAct cycle fits the other
four tools; this is the flow where that tradeoff flips.

State transitions:

    START -> assess -> route_by_amount
                            |-- amount <= threshold --> auto_approve -> END
                            '-- amount >  threshold --> await_decision -> route_by_decision
                                        |-- no human_decision yet --> END (paused)
                                        '-- human_decision present --> finalize -> END

The second branch is how the "pause" works in practice: the first
`ainvoke()` ends at `await_decision` with `status=pending_approval` and no
further edges to follow — the run simply stops, and `MemorySaver` persists
the state under `thread_id=application_id`. Approving/rejecting later means
calling `aupdate_state` to inject `human_decision`, then `ainvoke(None, ...)`
on the *same* thread_id — LangGraph resumes from the checkpoint, re-derives
the same route (this time `human_decision` is set), and reaches `finalize`.
See `services/loan_service.py` for both call sites.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from financial_agent.services.customer_service import CustomerService

_AUTO_APPROVE = "auto_approve"
_AWAIT_DECISION = "await_decision"
_FINALIZE = "finalize"


class LoanGraphState(TypedDict):
    application_id: str
    user_id: int
    amount: float
    status: str
    reason: str
    requires_human_approval: bool
    human_decision: Literal["approved", "rejected"] | None
    decided_by: str | None


def build_loan_graph(
    *,
    customer_service: CustomerService,
    approval_threshold: float,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph:
    """Compile the loan graph, closing over its collaborators (identity-bound, same
    spirit as the Tool factories in `agent/tools/*.py`: dependencies are wired in at
    construction time, never rediscovered from LLM-controlled input).

    `checkpointer` defaults to a fresh `MemorySaver` (handy for the unit tests in
    `tests/unit/test_loan_graph.py`, which don't care about the backend); production
    wiring (`api/app_state.py`) passes in the process-wide checkpointer shared with
    `agent/main_graph.py` — see that module's docstring for why this stays a separate
    graph rather than being folded into the main one.
    """

    async def assess(state: LoanGraphState) -> dict[str, object]:
        profile = await customer_service.get_profile(state["user_id"])
        return {
            "reason": (
                f"Assessment based on the customer's profile (segment {profile.segment}, "
                f"risk {profile.risk_profile.value})."
            )
        }

    def route_by_amount(state: LoanGraphState) -> str:
        return _AUTO_APPROVE if state["amount"] <= approval_threshold else _AWAIT_DECISION

    async def auto_approve(state: LoanGraphState) -> dict[str, object]:
        return {
            "status": "approved",
            "requires_human_approval": False,
            "reason": state["reason"] + " Amount within the auto-approval limit.",
        }

    async def await_decision(state: LoanGraphState) -> dict[str, object]:
        return {
            "status": "pending_approval",
            "requires_human_approval": True,
            "reason": (
                state["reason"] + " Amount above the auto-approval limit; awaiting human review."
            ),
        }

    def route_by_decision(state: LoanGraphState) -> str:
        return _FINALIZE if state.get("human_decision") else END

    async def finalize(state: LoanGraphState) -> dict[str, object]:
        approved = state.get("human_decision") == "approved"
        return {
            "status": "disbursed" if approved else "rejected",
            "reason": (
                "Approved in human review and disbursed."
                if approved
                else "Rejected in human review."
            ),
        }

    builder = StateGraph(LoanGraphState)
    builder.add_node("assess", assess)
    builder.add_node(_AUTO_APPROVE, auto_approve)
    builder.add_node(_AWAIT_DECISION, await_decision)
    builder.add_node(_FINALIZE, finalize)

    builder.set_entry_point("assess")
    builder.add_conditional_edges(
        "assess", route_by_amount, {_AUTO_APPROVE: _AUTO_APPROVE, _AWAIT_DECISION: _AWAIT_DECISION}
    )
    builder.add_edge(_AUTO_APPROVE, END)
    builder.add_conditional_edges(
        _AWAIT_DECISION, route_by_decision, {_FINALIZE: _FINALIZE, END: END}
    )
    builder.add_edge(_FINALIZE, END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
