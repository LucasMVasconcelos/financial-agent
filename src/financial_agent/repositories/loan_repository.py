"""Loan application repository.

A clean, queryable view of loan applications — separate from the LangGraph
checkpointer that actually drives the loan lifecycle (`agent/loan_graph.py`).
The checkpointer is keyed by `thread_id` and isn't meant to be listed or
filtered; this repository is what the admin approval endpoint queries
("show me pending applications") and what `LoanService` keeps in sync with
every graph transition.

`InMemoryLoanRepository` is process-local, same caveat as every other
`InMemory*` repository in this project: lost on restart, not shared across
replicas. A real deployment would back this with the same database as
`CustomerRepository`, since loan applications are exactly the kind of
record that needs durability and an audit trail independent of any LLM run.
"""

from __future__ import annotations

from typing import Protocol

from financial_agent.domain.models.loan import LoanApplication, LoanStatus


class LoanRepository(Protocol):
    async def save(self, application: LoanApplication) -> None:
        """Insert or overwrite the application (keyed by `application_id`)."""
        ...

    async def get_by_id(self, application_id: str) -> LoanApplication | None: ...

    async def list_by_status(self, status: LoanStatus) -> list[LoanApplication]: ...


class InMemoryLoanRepository:
    def __init__(self) -> None:
        self._applications: dict[str, LoanApplication] = {}

    async def save(self, application: LoanApplication) -> None:
        self._applications[application.application_id] = application

    async def get_by_id(self, application_id: str) -> LoanApplication | None:
        return self._applications.get(application_id)

    async def list_by_status(self, status: LoanStatus) -> list[LoanApplication]:
        return [a for a in self._applications.values() if a.status is status]
