"""Main conversational agent: prompt + tools + tool-calling AgentExecutor.

LangChain components used here, and why:

  * **Chat Model** (`llm_factory.build_chat_model`) — the underlying LLM,
    constructed once per process and reused across requests.
  * **PromptTemplate / System Prompt / Human Prompt** — `ChatPromptTemplate`
    composed of the versioned system prompt (`prompts/prompt_registry.py`),
    a `MessagesPlaceholder` for prior turns, the current human message, and
    a `MessagesPlaceholder("agent_scratchpad")` where the agent's
    intermediate tool calls/observations get injected.
  * **Tool Calling** — the five `StructuredTool`s from `agent/tools/`,
    bound per-request with the authenticated `user_id` baked in. Four are
    read-only lookups handled fine by this ReAct loop; `request_loan` is
    the one mutating action, and it deliberately does *not* try to express
    its human-approval wait inside this loop — it kicks off a separate,
    checkpointed LangGraph flow (`agent/loan_graph.py`) and returns
    immediately with a status. See that module's docstring for why.
  * **Output Parser** — handled internally by `create_openai_tools_agent`
    (see `agent/output_parser.py` docstring for the split with the filler
    chain).
  * **Runnable** — `create_openai_tools_agent` returns a `Runnable`; we
    don't invoke it directly, we hand it to...
  * **Agent Executor** — `AgentExecutor` is the loop that actually calls
    tools, feeds observations back to the LLM, and stops at a final answer
    or `max_iterations`. `handle_parsing_errors=True` means a malformed
    tool-call from the LLM becomes a retry turn instead of a crash — one
    more layer of "never let an exception interrupt the conversation".
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from financial_agent.agent.prompts.prompt_registry import get_active_system_prompt
from financial_agent.agent.tools import (
    build_get_customer_profile_tool,
    build_get_next_best_action_tool,
    build_get_products_tool,
    build_request_loan_tool,
    build_search_knowledge_base_tool,
)
from financial_agent.domain.models.conversation import ConversationHistory, MessageRole
from financial_agent.observability.logging import get_logger
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.knowledge_base_service import KnowledgeBaseService
from financial_agent.services.loan_service import LoanService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService

logger = get_logger(__name__)

_FALLBACK_REPLY = (
    "Desculpe, tive um problema para processar sua mensagem agora. "
    "Pode tentar novamente em instantes?"
)

_MAX_ITERATIONS = 6


def _to_langchain_messages(history: ConversationHistory) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in history.as_transcript():
        if turn.role is MessageRole.USER:
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    return messages


def build_agent_executor(
    *,
    user_id: int,
    llm: BaseChatModel,
    customer_service: CustomerService,
    nba_service: NBAService,
    products_service: ProductsService,
    knowledge_base_service: KnowledgeBaseService,
    loan_service: LoanService,
) -> AgentExecutor:
    """Assemble a request-scoped AgentExecutor with identity-bound tools."""
    tools = [
        build_get_next_best_action_tool(user_id=user_id, nba_service=nba_service),
        build_get_customer_profile_tool(user_id=user_id, customer_service=customer_service),
        build_get_products_tool(user_id=user_id, products_service=products_service),
        build_search_knowledge_base_tool(
            user_id=user_id, knowledge_base_service=knowledge_base_service
        ),
        build_request_loan_tool(user_id=user_id, loan_service=loan_service),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_active_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=_MAX_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


def _format_long_term_memories(memories: list[str]) -> str:
    if not memories:
        return "Nenhuma lembrança de longo prazo disponível ainda."
    return "\n".join(f"- {memory}" for memory in memories)


async def run_agent_turn(
    *,
    executor: AgentExecutor,
    user_message: str,
    history: ConversationHistory,
) -> str:
    """Run one conversation turn, guaranteeing a reply string is always returned."""
    try:
        result = await executor.ainvoke(
            {
                "input": user_message,
                "chat_history": _to_langchain_messages(history),
                "current_date": datetime.now(UTC).date().isoformat(),
                "conversation_summary": history.summary or "Nenhum resumo disponível ainda.",
                "long_term_memories": _format_long_term_memories(history.long_term_memories),
            }
        )
        output = result.get("output")
        if isinstance(output, str) and output.strip():
            return output
        logger.warning("agent_empty_output", user_id=history.user_id)
        return _FALLBACK_REPLY
    except Exception as exc:
        logger.error("agent_turn_failed", user_id=history.user_id, error=str(exc))
        return _FALLBACK_REPLY
