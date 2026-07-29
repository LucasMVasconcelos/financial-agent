from __future__ import annotations

import pytest

from financial_agent.domain.errors import ToolError as DomainToolError
from financial_agent.domain.errors import ToolErrorCode
from financial_agent.domain.models.customer import CustomerProfile
from financial_agent.domain.models.nba import NextBestActionCandidate, NextBestActionType
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.services.nba_service import NBAService


class _FailingGateway:
    def __init__(self, error: DomainToolError) -> None:
        self._error = error

    async def predict(self, customer: CustomerProfile) -> NextBestActionCandidate:
        raise self._error


class _StubGateway:
    async def predict(self, customer: CustomerProfile) -> NextBestActionCandidate:
        return NextBestActionCandidate(
            action=NextBestActionType.INVEST_CDB, confidence=0.9, reason="stub"
        )


class TestNBAService:
    async def test_happy_path(self) -> None:
        service = NBAService(InMemoryCustomerRepository(), _StubGateway())

        recommendation = await service.get_recommendation(123)

        assert recommendation.action == NextBestActionType.INVEST_CDB
        assert 0.0 <= recommendation.confidence <= 1.0

    async def test_not_found_when_customer_unknown(self) -> None:
        service = NBAService(InMemoryCustomerRepository(), _StubGateway())

        with pytest.raises(DomainToolError) as exc_info:
            await service.get_recommendation(999_999)

        assert exc_info.value.code == ToolErrorCode.NOT_FOUND

    async def test_upstream_error_propagates_from_gateway(self) -> None:
        gateway = _FailingGateway(DomainToolError(ToolErrorCode.UPSTREAM_ERROR, "model down"))
        service = NBAService(InMemoryCustomerRepository(), gateway)

        with pytest.raises(DomainToolError) as exc_info:
            await service.get_recommendation(123)

        assert exc_info.value.code == ToolErrorCode.UPSTREAM_ERROR

    async def test_rate_limited_propagates_from_gateway(self) -> None:
        gateway = _FailingGateway(DomainToolError(ToolErrorCode.RATE_LIMITED, "too many calls"))
        service = NBAService(InMemoryCustomerRepository(), gateway)

        with pytest.raises(DomainToolError) as exc_info:
            await service.get_recommendation(123)

        assert exc_info.value.code == ToolErrorCode.RATE_LIMITED
