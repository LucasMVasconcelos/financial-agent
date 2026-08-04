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

from financial_agent.agent.loan_graph import build_loan_graph
from financial_agent.agent.tools.get_customer_profile import build_get_customer_profile_tool
from financial_agent.agent.tools.get_next_best_action import build_get_next_best_action_tool
from financial_agent.agent.tools.get_products import build_get_products_tool
from financial_agent.agent.tools.request_loan import build_request_loan_tool
from financial_agent.agent.tools.search_knowledge_base import build_search_knowledge_base_tool
from financial_agent.domain.errors import ToolError as DomainToolError
from financial_agent.domain.errors import ToolErrorCode
from financial_agent.domain.models.knowledge import KnowledgeSnippet
from financial_agent.gateways.nba_model_gateway import MockNBAModelGateway
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.repositories.loan_repository import InMemoryLoanRepository
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.knowledge_base_service import KnowledgeBaseService
from financial_agent.services.loan_service import LoanService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService

KNOWN_USER_ID = 123
UNKNOWN_USER_ID = 999_999


class _NullTelegramGateway:
    async def send_message(self, *, chat_id: int, text: str) -> None:
        return None


class _StubKnowledgeBaseGateway:
    def __init__(
        self,
        *,
        snippets: list[KnowledgeSnippet] | None = None,
        error: DomainToolError | None = None,
    ) -> None:
        self._snippets = snippets or []
        self._error = error

    async def search(self, query: str, *, top_k: int = 3) -> list[KnowledgeSnippet]:
        if self._error is not None:
            raise self._error
        return self._snippets


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


class TestSearchKnowledgeBaseToolContract:
    def _build(
        self,
        *,
        snippets: list[KnowledgeSnippet] | None = None,
        error: DomainToolError | None = None,
    ) -> object:
        gateway = _StubKnowledgeBaseGateway(snippets=snippets, error=error)
        return build_search_knowledge_base_tool(
            user_id=KNOWN_USER_ID, knowledge_base_service=KnowledgeBaseService(gateway)
        )

    def test_input_schema_only_exposes_query(self) -> None:
        tool = self._build()
        assert set(tool.args_schema.model_fields.keys()) == {"query"}

    async def test_success_envelope_matches_output_schema(self) -> None:
        snippet = KnowledgeSnippet(
            title="Tesouro Selic",
            content="Floating-rate government bond that tracks the Selic rate.",
            source="tesouro_selic",
            score=0.87,
        )
        tool = self._build(snippets=[snippet])

        raw = await tool.ainvoke({"query": "how does tesouro selic work"})
        envelope = json.loads(raw)

        assert envelope["success"] is True
        result = envelope["data"]["results"][0]
        assert set(result.keys()) == {"title", "content", "source", "score"}
        assert result["source"] == "tesouro_selic"

    async def test_upstream_error_structured(self) -> None:
        tool = self._build(error=DomainToolError(ToolErrorCode.UPSTREAM_ERROR, "vector store down"))

        raw = await tool.ainvoke({"query": "tesouro selic"})
        envelope = json.loads(raw)

        assert envelope["success"] is False
        assert envelope["error"]["code"] == ToolErrorCode.UPSTREAM_ERROR.value


class TestRequestLoanToolContract:
    def _build(self, *, user_id: int) -> object:
        repo = InMemoryCustomerRepository()
        customer_service = CustomerService(repo)
        graph = build_loan_graph(customer_service=customer_service, approval_threshold=50_000.0)
        loan_service = LoanService(
            graph=graph,
            loan_repository=InMemoryLoanRepository(),
            telegram_gateway=_NullTelegramGateway(),
        )
        return build_request_loan_tool(user_id=user_id, loan_service=loan_service)

    def test_input_schema_only_exposes_amount(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)
        assert set(tool.args_schema.model_fields.keys()) == {"amount"}

    async def test_success_envelope_matches_output_schema(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)

        raw = await tool.ainvoke({"amount": 1000.0})
        envelope = json.loads(raw)

        assert envelope["success"] is True
        assert set(envelope["data"].keys()) == {
            "application_id",
            "status",
            "reason",
            "requires_human_approval",
        }

    async def test_small_amount_is_approved_without_human_review(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)

        raw = await tool.ainvoke({"amount": 1000.0})
        envelope = json.loads(raw)

        assert envelope["data"]["status"] == "approved"
        assert envelope["data"]["requires_human_approval"] is False

    async def test_large_amount_requires_human_approval(self) -> None:
        tool = self._build(user_id=KNOWN_USER_ID)

        raw = await tool.ainvoke({"amount": 80_000.0})
        envelope = json.loads(raw)

        assert envelope["data"]["status"] == "pending_approval"
        assert envelope["data"]["requires_human_approval"] is True

    async def test_not_found_structured_error(self) -> None:
        tool = self._build(user_id=UNKNOWN_USER_ID)

        raw = await tool.ainvoke({"amount": 1000.0})
        envelope = json.loads(raw)

        assert envelope["success"] is False
        assert envelope["error"]["code"] == ToolErrorCode.NOT_FOUND.value
