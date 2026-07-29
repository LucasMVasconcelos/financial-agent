"""Tool Contract Tests.

For every Tool exposed to the LLM, verify:
  * Input Schema — no identity field (`user_id`) is ever LLM-controllable.
  * Output Schema / Serialization — a successful call returns a JSON string
    that round-trips into the documented `ToolEnvelope` shape.
  * Structured Errors — a failure path returns `success: false` with one of
    the documented `ToolErrorCode`s, never a raw exception.
"""

from __future__ import annotations

import json

from financial_agent.agent.tools.get_customer_profile import build_get_customer_profile_tool
from financial_agent.agent.tools.get_next_best_action import build_get_next_best_action_tool
from financial_agent.agent.tools.get_products import build_get_products_tool
from financial_agent.domain.errors import ToolErrorCode
from financial_agent.gateways.nba_model_gateway import MockNBAModelGateway
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService

KNOWN_USER_ID = 123
UNKNOWN_USER_ID = 999_999


class TestGetNextBestActionToolContract:
    def _build(self, *, user_id: int) -> object:
        repo = InMemoryCustomerRepository()
        return build_get_next_best_action_tool(
            user_id=user_id, nba_service=NBAService(repo, MockNBAModelGateway())
        )

    def test_input_schema_has_no_identity_field(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)
        assert "user_id" not in tool.args_schema.model_fields

    async def test_success_envelope_matches_output_schema(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)

        raw = await tool.ainvoke({})
        envelope = json.loads(raw)

        assert envelope["success"] is True
        assert set(envelope["data"].keys()) == {"action", "confidence", "reason"}
        assert 0.0 <= envelope["data"]["confidence"] <= 1.0

    async def test_not_found_structured_error(self) -> None:
        tool = self._build(user_id=UNKNOWN_USER_ID)

        raw = await tool.ainvoke({})
        envelope = json.loads(raw)

        assert envelope["success"] is False
        assert envelope["error"]["code"] == ToolErrorCode.NOT_FOUND.value


class TestGetCustomerProfileToolContract:
    def _build(self, *, user_id: int) -> object:
        repo = InMemoryCustomerRepository()
        return build_get_customer_profile_tool(
            user_id=user_id, customer_service=CustomerService(repo)
        )

    def test_input_schema_has_no_identity_field(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)
        assert "user_id" not in tool.args_schema.model_fields

    async def test_success_envelope_matches_output_schema(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)

        raw = await tool.ainvoke({})
        envelope = json.loads(raw)

        assert envelope["success"] is True
        assert set(envelope["data"].keys()) == {
            "full_name",
            "segment",
            "risk_profile",
            "account_balance",
            "owned_products",
        }

    async def test_not_found_structured_error(self) -> None:
        tool = self._build(user_id=UNKNOWN_USER_ID)

        raw = await tool.ainvoke({})
        envelope = json.loads(raw)

        assert envelope["success"] is False
        assert envelope["error"]["code"] == ToolErrorCode.NOT_FOUND.value


class TestGetProductsToolContract:
    def _build(self, *, user_id: int) -> object:
        repo = InMemoryCustomerRepository()
        return build_get_products_tool(user_id=user_id, products_service=ProductsService(repo))

    def test_input_schema_only_exposes_category_filter(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)
        assert set(tool.args_schema.model_fields.keys()) == {"category"}

    async def test_success_envelope_matches_output_schema(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)

        raw = await tool.ainvoke({})
        envelope = json.loads(raw)

        assert envelope["success"] is True
        product = envelope["data"]["products"][0]
        assert set(product.keys()) == {"code", "name", "category", "owned_by_customer"}

    async def test_category_filter_narrows_results(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)

        raw = await tool.ainvoke({"category": "seguro"})
        envelope = json.loads(raw)

        assert envelope["success"] is True
        assert all(p["category"] == "seguro" for p in envelope["data"]["products"])

    async def test_not_found_structured_error(self) -> None:
        tool = self._build(user_id=UNKNOWN_USER_ID)

        raw = await tool.ainvoke({})
        envelope = json.loads(raw)

        assert envelope["success"] is False
        assert envelope["error"]["code"] == ToolErrorCode.NOT_FOUND.value
