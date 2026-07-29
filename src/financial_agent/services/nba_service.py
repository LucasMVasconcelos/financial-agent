"""Next Best Action service.

Orchestrates: load the customer profile -> call the (mocked or real) NBA
model gateway -> return a candidate recommendation. Kept separate from
`CustomerService` because the two have independent failure modes and, in a
real deployment, likely different SLAs/timeouts (the model call is the one
worth putting a circuit breaker / timeout / cache around).
"""

from __future__ import annotations

from financial_agent.domain.errors import ToolError, ToolErrorCode
from financial_agent.domain.models.nba import NextBestActionCandidate
from financial_agent.gateways.nba_model_gateway import NBAModelGateway
from financial_agent.repositories.customer_repository import CustomerRepository


class NBAService:
    def __init__(
        self,
        customer_repository: CustomerRepository,
        nba_model_gateway: NBAModelGateway,
    ) -> None:
        self._customer_repository = customer_repository
        self._nba_model_gateway = nba_model_gateway

    async def get_recommendation(self, user_id: int) -> NextBestActionCandidate:
        profile = await self._customer_repository.get_by_id(user_id)
        if profile is None:
            raise ToolError(
                ToolErrorCode.NOT_FOUND,
                "Cannot compute a recommendation: customer not found.",
                details={"user_id": user_id},
            )
        return await self._nba_model_gateway.predict(profile)
