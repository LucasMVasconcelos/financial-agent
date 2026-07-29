"""Customer repository: abstracts *where* customer data lives.

Only the interface (`CustomerRepository`) should be imported by services.
`InMemoryCustomerRepository` is a seeded fake standing in for a real
datastore (Postgres, a CRM API, a feature store, ...) — swapping it for a
real implementation later is a matter of implementing the same Protocol and
changing the wiring in `api/deps.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from financial_agent.domain.models.customer import CustomerProfile, Product, RiskProfile


class CustomerRepository(Protocol):
    async def get_by_id(self, user_id: int) -> CustomerProfile | None:
        """Return the customer profile for `user_id`, or None if unknown."""
        ...


class InMemoryCustomerRepository:
    """Seeded in-memory fake, keyed by Telegram user id."""

    def __init__(self) -> None:
        self._customers: dict[int, CustomerProfile] = {
            123: CustomerProfile(
                user_id=123,
                full_name="Ana Souza",
                segment="high_income",
                risk_profile=RiskProfile.MODERATE,
                account_balance=85_000.00,
                products=[
                    Product(
                        code="conta_corrente",
                        name="Conta Corrente",
                        category="conta",
                        owned_by_customer=True,
                    ),
                    Product(
                        code="cartao_black",
                        name="Cartão Black",
                        category="cartao",
                        owned_by_customer=True,
                    ),
                ],
                created_at=datetime(2021, 3, 15, tzinfo=UTC),
            ),
            456: CustomerProfile(
                user_id=456,
                full_name="Bruno Lima",
                segment="varejo",
                risk_profile=RiskProfile.CONSERVATIVE,
                account_balance=1_200.50,
                products=[
                    Product(
                        code="conta_corrente",
                        name="Conta Corrente",
                        category="conta",
                        owned_by_customer=True,
                    ),
                ],
                created_at=datetime(2023, 7, 2, tzinfo=UTC),
            ),
        }

    async def get_by_id(self, user_id: int) -> CustomerProfile | None:
        return self._customers.get(user_id)
