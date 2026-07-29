from __future__ import annotations

import pytest

from financial_agent.domain.errors import ToolError, ToolErrorCode
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.services.customer_service import CustomerService


@pytest.fixture
def service() -> CustomerService:
    return CustomerService(InMemoryCustomerRepository())


class TestCustomerService:
    async def test_happy_path_returns_profile(self, service: CustomerService) -> None:
        profile = await service.get_profile(123)

        assert profile.user_id == 123
        assert profile.full_name == "Ana Souza"

    async def test_not_found_raises_structured_error(self, service: CustomerService) -> None:
        with pytest.raises(ToolError) as exc_info:
            await service.get_profile(999_999)

        assert exc_info.value.code == ToolErrorCode.NOT_FOUND
        assert exc_info.value.details == {"user_id": 999_999}
