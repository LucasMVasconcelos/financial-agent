"""Customer profile service.

Thin orchestration layer between the repository and the Tool handlers:
translates "not found" into the shared `ToolError` contract so every caller
(Tool, future REST endpoint, ...) gets the same structured error shape.
"""

from __future__ import annotations

from financial_agent.domain.errors import ToolError, ToolErrorCode
from financial_agent.domain.models.customer import CustomerProfile
from financial_agent.repositories.customer_repository import CustomerRepository


class CustomerService:
    def __init__(self, customer_repository: CustomerRepository) -> None:
        self._customer_repository = customer_repository

    async def get_profile(self, user_id: int) -> CustomerProfile:
        profile = await self._customer_repository.get_by_id(user_id)
        if profile is None:
            raise ToolError(
                ToolErrorCode.NOT_FOUND,
                "No customer profile found for this user.",
                details={"user_id": user_id},
            )
        return profile
