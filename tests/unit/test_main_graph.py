"""Unit tests for the main conversational graph (agent/main_graph.py).

Exercises the cyclic FSM mechanics directly against `MemorySaver` (no
network, no real LLM): the ReAct loop (reason -> tool -> reason), the
deterministic validation gate rejecting a bad tool call before it ever
reaches a service, Tool Self-Correction looping back to `reason` on a
structured tool error, the bounded retry cap, and checkpointing. The LLM is
replaced by `_ScriptedChatModel`, which plays back a fixed script of
`AIMessage`s (including `tool_calls`) instead of calling OpenAI — the same
role `FakeListChatModel` plays elsewhere in this suite, but preserving
`tool_calls` requires scripting whole messages instead of response strings.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from financial_agent.agent.loan_graph import build_loan_graph
from financial_agent.agent.main_graph import build_main_graph, run_main_graph_turn
from financial_agent.domain.models.conversation import ConversationHistory
from financial_agent.domain.models.knowledge import KnowledgeSnippet
from financial_agent.domain.models.nba import NextBestActionCandidate, NextBestActionType
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.repositories.loan_repository import InMemoryLoanRepository
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.knowledge_base_service import KnowledgeBaseService
from financial_agent.services.loan_service import LoanService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService

KNOWN_USER_ID = 123
UNKNOWN_USER_ID = 999_999


class _ScriptedChatModel(FakeMessagesListChatModel):
    """Same behavior as `FakeMessagesListChatModel`, minus the parent's
    `bind_tools` raising `NotImplementedError` — this graph always binds
    tools onto the router-selected model before invoking it."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedChatModel:
        return self


class _StubNBAGateway:
    async def predict(self, customer: Any) -> NextBestActionCandidate:
        return NextBestActionCandidate(
            action=NextBestActionType.INVEST_CDB, confidence=0.9, reason="stub"
        )


class _StubKnowledgeBaseGateway:
    async def search(self, query: str, *, top_k: int = 3) -> list[KnowledgeSnippet]:
        return []


class _StubTelegramGateway:
    async def send_message(self, *, chat_id: int, text: str) -> None:
        return None


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _repeated_tool_call_messages(name: str, args: dict[str, Any], count: int) -> list[AIMessage]:
    """`count` distinct `AIMessage`s, each calling the same tool.

    Deliberately distinct objects (not one message reused): `add_messages`
    dedups by message `.id`, and a real LLM never returns the same message id
    twice — reusing one `AIMessage` instance across scripted turns would
    silently collapse them into a single list entry instead of the growing
    transcript a real multi-turn ReAct loop produces.
    """
    return [
        AIMessage(content="", tool_calls=[_tool_call(name, args, f"call_{i}")])
        for i in range(count)
    ]


def _build_dependencies() -> dict[str, Any]:
    customer_repository = InMemoryCustomerRepository()
    customer_service = CustomerService(customer_repository)
    loan_graph = build_loan_graph(customer_service=customer_service, approval_threshold=50_000.0)
    loan_service = LoanService(
        graph=loan_graph,
        loan_repository=InMemoryLoanRepository(),
        telegram_gateway=_StubTelegramGateway(),
    )
    return {
        "customer_service": customer_service,
        "nba_service": NBAService(customer_repository, _StubNBAGateway()),
        "products_service": ProductsService(customer_repository),
        "knowledge_base_service": KnowledgeBaseService(_StubKnowledgeBaseGateway()),
        "loan_service": loan_service,
    }


def _compile_graph(
    *, responses: list[AIMessage], user_id: int = KNOWN_USER_ID
) -> CompiledStateGraph:
    return build_main_graph(
        user_id=user_id,
        llm=_ScriptedChatModel(responses=responses),
        checkpointer=MemorySaver(),
        **_build_dependencies(),
    )


def _initial_state(user_id: int = KNOWN_USER_ID) -> dict[str, Any]:
    return {
        "messages": [("human", "Olá")],
        "user_id": user_id,
        "iteration": 0,
        "tool_retry_count": 0,
        "final_response": None,
    }


class TestMainGraphReactCycle:
    async def test_no_tool_call_finalizes_in_one_pass(self) -> None:
        graph = _compile_graph(responses=[AIMessage(content="Olá! Como posso ajudar?")])
        config = {"configurable": {"thread_id": "t-1"}}

        result = await graph.ainvoke(_initial_state(), config)

        assert result["final_response"] == "Olá! Como posso ajudar?"
        assert result["iteration"] == 1

    async def test_tool_call_then_final_answer_is_a_real_cycle(self) -> None:
        graph = _compile_graph(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_customer_profile", {}, "call_1")],
                ),
                AIMessage(content="Seu perfil está ótimo!"),
            ]
        )
        config = {"configurable": {"thread_id": "t-2"}}

        result = await graph.ainvoke(_initial_state(), config)

        assert result["final_response"] == "Seu perfil está ótimo!"
        # Two visits to `reason` is the cycle: reason -> tool -> reason -> finalize.
        assert result["iteration"] == 2
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert '"success":true' in tool_messages[0].content

    async def test_iteration_cap_forces_finalize(self) -> None:
        # Always calls a tool — without the cap this would cycle forever.
        graph = _compile_graph(
            responses=_repeated_tool_call_messages("get_customer_profile", {}, 6)
        )
        config = {"configurable": {"thread_id": "t-3"}}

        result = await graph.ainvoke(_initial_state(), config)

        assert result["iteration"] == 6  # _MAX_ITERATIONS
        assert result["final_response"]  # falls back to the default reply text


class TestMainGraphDeterministicValidation:
    async def test_unknown_tool_name_is_rejected_before_execution(self) -> None:
        graph = _compile_graph(
            responses=[
                AIMessage(content="", tool_calls=[_tool_call("delete_everything", {}, "call_bad")]),
                AIMessage(content="Não posso fazer isso, mas posso ajudar de outra forma."),
            ]
        )
        config = {"configurable": {"thread_id": "t-4"}}

        result = await graph.ainvoke(_initial_state(), config)

        assert result["final_response"] == "Não posso fazer isso, mas posso ajudar de outra forma."
        assert result["tool_retry_count"] == 1
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert '"success":false' in tool_messages[0].content
        assert "VALIDATION_ERROR" in tool_messages[0].content

    async def test_out_of_range_loan_amount_is_rejected_before_execution(self) -> None:
        graph = _compile_graph(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("request_loan", {"amount": -5}, "call_bad_amount")],
                ),
                AIMessage(content="Não é possível solicitar esse valor."),
            ]
        )
        config = {"configurable": {"thread_id": "t-5"}}

        result = await graph.ainvoke(_initial_state(), config)

        assert result["final_response"] == "Não é possível solicitar esse valor."
        assert result["tool_retry_count"] == 1


class TestMainGraphToolSelfCorrection:
    async def test_structured_tool_error_triggers_self_correction_and_recovers(self) -> None:
        graph = _compile_graph(
            responses=[
                AIMessage(
                    content="", tool_calls=[_tool_call("get_customer_profile", {}, "call_1")]
                ),
                AIMessage(content="Não encontrei seu cadastro, mas posso ajudar de outra forma."),
            ],
            user_id=UNKNOWN_USER_ID,
        )
        config = {"configurable": {"thread_id": "t-6"}}

        result = await graph.ainvoke(_initial_state(user_id=UNKNOWN_USER_ID), config)

        assert result["final_response"] == (
            "Não encontrei seu cadastro, mas posso ajudar de outra forma."
        )
        assert result["tool_retry_count"] == 1
        assert result["iteration"] == 2
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert '"success":false' in tool_messages[0].content
        assert "NOT_FOUND" in tool_messages[0].content

    async def test_retries_are_bounded(self) -> None:
        # Every reasoning step calls the same failing tool — self_correct must
        # stop looping once `_MAX_TOOL_RETRIES` is hit instead of retrying forever.
        graph = _compile_graph(
            responses=_repeated_tool_call_messages("get_customer_profile", {}, 6),
            user_id=UNKNOWN_USER_ID,
        )
        config = {"configurable": {"thread_id": "t-7"}}

        result = await graph.ainvoke(_initial_state(user_id=UNKNOWN_USER_ID), config)

        assert result["tool_retry_count"] == 2  # _MAX_TOOL_RETRIES
        assert result["iteration"] == 6  # eventually hits the iteration cap too


class TestMainGraphCheckpointing:
    async def test_state_is_checkpointed_per_thread_id(self) -> None:
        graph = _compile_graph(responses=[AIMessage(content="Oi!")])
        config = {"configurable": {"thread_id": "t-checkpoint"}}

        await graph.ainvoke(_initial_state(), config)
        snapshot = await graph.aget_state(config)

        assert snapshot.values["final_response"] == "Oi!"

    async def test_different_threads_do_not_share_state(self) -> None:
        graph = _compile_graph(
            responses=[
                AIMessage(content="", tool_calls=[_tool_call("get_customer_profile", {}, "c1")]),
                AIMessage(content="resposta A"),
            ]
        )
        cfg_a = {"configurable": {"thread_id": "t-a"}}
        cfg_b = {"configurable": {"thread_id": "t-b"}}

        await graph.ainvoke(_initial_state(), cfg_a)
        state_a = await graph.aget_state(cfg_a)
        state_b = await graph.aget_state(cfg_b)

        assert state_a.values["final_response"] == "resposta A"
        assert state_b.values == {}  # thread "t-b" was never invoked


class TestRunMainGraphTurn:
    async def test_returns_final_response_from_the_graph(self) -> None:
        graph = _compile_graph(responses=[AIMessage(content="Recomendamos investir em CDB.")])

        reply = await run_main_graph_turn(
            graph=graph,
            thread_id="turn-1",
            user_id=KNOWN_USER_ID,
            user_message="O que você recomenda?",
            history=ConversationHistory(user_id=KNOWN_USER_ID),
        )

        assert reply == "Recomendamos investir em CDB."

    async def test_graph_exception_returns_fallback_reply(self) -> None:
        class _BrokenGraph:
            async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("boom")

        reply = await run_main_graph_turn(
            graph=_BrokenGraph(),  # type: ignore[arg-type]
            thread_id="turn-2",
            user_id=KNOWN_USER_ID,
            user_message="oi",
            history=ConversationHistory(user_id=KNOWN_USER_ID),
        )

        assert "problema" in reply.lower()
