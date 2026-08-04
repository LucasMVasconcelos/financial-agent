"""Main conversational agent — a cyclic, checkpointed LangGraph FSM.

Replaces `agent_executor.py`'s `AgentExecutor`/`create_openai_tools_agent`
loop with an explicit `StateGraph`. The behavior an outside caller sees is
the same (send a message, get a reply, tools get called along the way) but
every step the old opaque `AgentExecutor` loop performed implicitly is now
its own node, individually observable, checkpointed, and retryable:

    START -> reason -[tool_calls?]-> validate_tool_calls -[all valid?]-> execute_tool
               ^                            |  (invalid)                     |
               |                            v                                v
               |                       self_correct <-----[any tool failed]--+
               |                            |
               +----------------------------+
               |
               '-[no tool_calls, or iteration cap hit]-> finalize -> END

  * **reason** — the ReAct step: the LLM sees the running `messages` list
    (system prompt, history, prior tool results) and either calls a tool or
    produces a final answer. This node is visited on every loop iteration,
    which is what makes the graph cyclic rather than a fixed pipeline.
  * **validate_tool_calls** — deterministic validation *between* steps, run
    before any tool executes: unknown tool name, an identity field smuggled
    into arguments (defense in depth — narrow schemas already exclude
    `user_id`, this is a second gate against a future schema regression),
    or an out-of-range `request_loan` amount. A failure here never touches
    a service — it short-circuits straight to `self_correct` with a
    synthetic `ToolEnvelope` error, so the LLM sees exactly the same
    structured-error shape a real tool failure would produce.
  * **execute_tool** — runs the validated tool call(s) and appends their
    `ToolMessage` results. Tools already guarantee (see `agent/tools/base.py`
    `run_tool`) that no exception escapes here; every result is a
    `ToolEnvelope` JSON string.
  * **self_correct** — Tool Self-Correction: on a structured tool error
    (validation failure or `ToolEnvelope.success=False`), inject a
    corrective instruction and loop back to `reason` instead of giving up
    or crashing the turn. Bounded by `_MAX_TOOL_RETRIES` so a
    persistently-failing tool degrades to `finalize`'s fallback reply
    rather than looping forever.
  * **finalize** — extracts the final assistant text once `reason` produces
    a plain answer (no more tool calls) or the iteration cap is hit.

Human-in-the-loop for this project lives in the *loan* subgraph
(`agent/loan_graph.py`), not here. `request_loan` is a Tool like any
other — it returns immediately with a status (`pending_approval` included)
and this graph never blocks a turn waiting on a human. Pausing an entire
conversation turn for hours would be poor UX (and pointless: this graph's
checkpoint only needs to survive one Telegram round-trip); the loan
subgraph's job is exactly the "pause for a human, resume in a wholly
separate flow" case, and it now shares this graph's Redis checkpointer
backend (see `api/app_state.py`) — the whole project runs on LangGraph, but
as two purpose-built graphs rather than one, because the two have genuinely
different pause semantics.

Persistence: every node transition is checkpointed under
`thread_id=<per-turn id>` via the injected checkpointer (`RedisSaver` in
production, `MemorySaver` in tests — see `api/app_state.py`), giving crash
recovery mid-turn "for free": a process restart between `execute_tool` and
`reason` resumes from the last completed node instead of losing the turn.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from financial_agent.agent.prompts.prompt_registry import get_active_system_prompt
from financial_agent.agent.tools import (
    build_get_customer_profile_tool,
    build_get_next_best_action_tool,
    build_get_products_tool,
    build_request_loan_tool,
    build_search_knowledge_base_tool,
)
from financial_agent.agent.tools.base import ToolEnvelope
from financial_agent.agent.tools.request_loan import REQUEST_LOAN_TOOL_NAME
from financial_agent.domain.errors import ToolError as DomainToolError
from financial_agent.domain.errors import ToolErrorCode
from financial_agent.domain.models.conversation import ConversationHistory, MessageRole
from financial_agent.observability.logging import get_logger
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.knowledge_base_service import KnowledgeBaseService
from financial_agent.services.loan_service import LoanService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService

logger = get_logger(__name__)

_FALLBACK_REPLY = (
    "Sorry, I had a problem processing your message just now. " "Could you try again in a moment?"
)

_MAX_ITERATIONS = 6
_MAX_TOOL_RETRIES = 2

_REASON = "reason"
_VALIDATE_TOOL_CALLS = "validate_tool_calls"
_EXECUTE_TOOL = "execute_tool"
_SELF_CORRECT = "self_correct"
_FINALIZE = "finalize"


class MainGraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: int
    iteration: int
    tool_retry_count: int
    final_response: str | None


def _validation_error_envelope(message: str) -> str:
    """Same `ToolEnvelope` shape a real tool failure produces (see
    `agent/tools/base.py` `run_tool`), so the LLM can't tell a deterministic
    pre-execution rejection apart from a downstream service error."""
    envelope: ToolEnvelope[Any] = ToolEnvelope.fail(
        DomainToolError(ToolErrorCode.VALIDATION_ERROR, message)
    )
    return envelope.model_dump_json()


def _tool_call_succeeded(message: ToolMessage) -> bool:
    """Every tool result is a `ToolEnvelope` JSON string (see `run_tool` and
    `_validation_error_envelope`) — `content` is typed as `str | list[...]`
    on the base class, but never anything but `str` in this project."""
    content = message.content
    assert isinstance(content, str)
    return bool(json.loads(content).get("success", True))


_MAX_LOAN_AMOUNT = 10_000_000


def _validate_tool_call(call: ToolCall, tools_by_name: dict[str, StructuredTool]) -> str | None:
    """Deterministic gate run before any tool executes. Returns an error message,
    or None if the call is allowed to proceed."""
    name = call["name"]
    if name not in tools_by_name:
        return f"Unknown tool '{name}'."
    args = call.get("args") or {}
    if "user_id" in args:
        # Defense in depth: narrow schemas never declare this field, so this
        # should be unreachable — but if a future schema regresses, refusing
        # here is cheaper than trusting an LLM-controlled identity override.
        return "Tool arguments must never include an identity field."
    if name == REQUEST_LOAN_TOOL_NAME:
        amount = args.get("amount")
        if not isinstance(amount, int | float) or amount <= 0 or amount > _MAX_LOAN_AMOUNT:
            return "Loan amount is missing or out of the allowed range."
    return None


def _to_langchain_messages(history: ConversationHistory) -> list[AnyMessage]:
    messages: list[AnyMessage] = []
    for turn in history.as_transcript():
        if turn.role is MessageRole.USER:
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    return messages


def _format_long_term_memories(memories: list[str]) -> str:
    if not memories:
        return "No long-term memories available yet."
    return "\n".join(f"- {memory}" for memory in memories)


def _build_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", get_active_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_main_graph(
    *,
    user_id: int,
    llm: BaseChatModel,
    customer_service: CustomerService,
    nba_service: NBAService,
    products_service: ProductsService,
    knowledge_base_service: KnowledgeBaseService,
    loan_service: LoanService,
    checkpointer: BaseCheckpointSaver[str],
) -> CompiledStateGraph:
    """Compile a request-scoped main graph with identity-bound tools.

    Same identity-required-dispatch spirit as every Tool factory in
    `agent/tools/*.py`: tools close over the authenticated `user_id` at construction time,
    so the graph is rebuilt per request even though it shares one
    process-wide `checkpointer` instance across every user (thread_ids are
    per-turn and never collide).
    """
    tools = [
        build_get_next_best_action_tool(user_id=user_id, nba_service=nba_service),
        build_get_customer_profile_tool(user_id=user_id, customer_service=customer_service),
        build_get_products_tool(user_id=user_id, products_service=products_service),
        build_search_knowledge_base_tool(
            user_id=user_id, knowledge_base_service=knowledge_base_service
        ),
        build_request_loan_tool(user_id=user_id, loan_service=loan_service),
    ]
    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    async def reason(state: MainGraphState) -> dict[str, Any]:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response], "iteration": state["iteration"] + 1}

    def route_after_reason(state: MainGraphState) -> str:
        last = state["messages"][-1]
        has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
        if has_tool_calls and state["iteration"] < _MAX_ITERATIONS:
            return _VALIDATE_TOOL_CALLS
        return _FINALIZE

    async def validate_tool_calls(state: MainGraphState) -> dict[str, Any] | None:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        errors = {call["id"]: _validate_tool_call(call, tools_by_name) for call in last.tool_calls}
        if not any(errors.values()):
            return None
        logger.warning(
            "tool_call_validation_failed",
            user_id=user_id,
            errors={k: v for k, v in errors.items() if v},
        )
        rejections = [
            ToolMessage(
                content=_validation_error_envelope(errors[call["id"]] or "Invalid tool call."),
                tool_call_id=call["id"],
                name=call["name"],
            )
            for call in last.tool_calls
        ]
        return {"messages": rejections}

    def route_after_validate(state: MainGraphState) -> str:
        last = state["messages"][-1]
        return _SELF_CORRECT if isinstance(last, ToolMessage) else _EXECUTE_TOOL

    async def execute_tool(state: MainGraphState) -> dict[str, Any]:
        last_ai_message = next(
            m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.tool_calls
        )
        results = []
        for call in last_ai_message.tool_calls:
            tool = tools_by_name[call["name"]]
            raw = await tool.ainvoke(call["args"])
            results.append(ToolMessage(content=raw, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    def _recent_tool_messages(state: MainGraphState) -> list[ToolMessage]:
        messages = state["messages"]
        cutoff = next(
            i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], AIMessage)
        )
        return [m for m in messages[cutoff + 1 :] if isinstance(m, ToolMessage)]

    def route_after_execute(state: MainGraphState) -> str:
        if state["tool_retry_count"] >= _MAX_TOOL_RETRIES:
            return _REASON
        any_failed = any(not _tool_call_succeeded(m) for m in _recent_tool_messages(state))
        return _SELF_CORRECT if any_failed else _REASON

    async def self_correct(state: MainGraphState) -> dict[str, Any]:
        correction = HumanMessage(
            content=(
                "One or more tools returned a structured error (see the 'error' "
                "field above). Review the reason and try again with a corrected "
                "approach (different arguments, a different tool), or, if there is "
                "no way to fix it, explain that to the customer instead of "
                "insisting."
            )
        )
        return {"messages": [correction], "tool_retry_count": state["tool_retry_count"] + 1}

    async def finalize(state: MainGraphState) -> dict[str, Any]:
        last = state["messages"][-1]
        text = last.content if isinstance(last, AIMessage) and isinstance(last.content, str) else ""
        return {"final_response": text.strip() or _FALLBACK_REPLY}

    builder = StateGraph(MainGraphState)
    builder.add_node(_REASON, reason)
    builder.add_node(_VALIDATE_TOOL_CALLS, validate_tool_calls)
    builder.add_node(_EXECUTE_TOOL, execute_tool)
    builder.add_node(_SELF_CORRECT, self_correct)
    builder.add_node(_FINALIZE, finalize)

    builder.set_entry_point(_REASON)
    builder.add_conditional_edges(
        _REASON,
        route_after_reason,
        {_VALIDATE_TOOL_CALLS: _VALIDATE_TOOL_CALLS, _FINALIZE: _FINALIZE},
    )
    builder.add_conditional_edges(
        _VALIDATE_TOOL_CALLS,
        route_after_validate,
        {_EXECUTE_TOOL: _EXECUTE_TOOL, _SELF_CORRECT: _SELF_CORRECT},
    )
    builder.add_conditional_edges(
        _EXECUTE_TOOL, route_after_execute, {_REASON: _REASON, _SELF_CORRECT: _SELF_CORRECT}
    )
    builder.add_edge(_SELF_CORRECT, _REASON)
    builder.add_edge(_FINALIZE, END)

    return builder.compile(checkpointer=checkpointer)


async def run_main_graph_turn(
    *,
    graph: CompiledStateGraph,
    thread_id: str,
    user_id: int,
    user_message: str,
    history: ConversationHistory,
) -> str:
    """Run one conversation turn, guaranteeing a reply string is always returned."""
    prompt = _build_prompt_template()
    initial_messages = cast(
        "list[AnyMessage]",
        prompt.invoke(
            {
                "chat_history": _to_langchain_messages(history),
                "input": user_message,
                "current_date": datetime.now(UTC).date().isoformat(),
                "conversation_summary": history.summary or "No summary available yet.",
                "long_term_memories": _format_long_term_memories(history.long_term_memories),
            }
        ).to_messages(),
    )

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    initial_state: MainGraphState = {
        "messages": initial_messages,
        "user_id": user_id,
        "iteration": 0,
        "tool_retry_count": 0,
        "final_response": None,
    }
    try:
        result = await graph.ainvoke(initial_state, config)
        output = result.get("final_response")
        if isinstance(output, str) and output.strip():
            return output
        logger.warning("agent_empty_output", user_id=user_id)
        return _FALLBACK_REPLY
    except Exception as exc:
        logger.error("agent_turn_failed", user_id=user_id, error=str(exc))
        return _FALLBACK_REPLY
