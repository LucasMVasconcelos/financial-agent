"""Filler agent: keeps the conversation warm while the main agent works.

The main agent (`agent/main_graph.py`) may take a few seconds per turn —
it calls `get_customer_profile` and `get_next_best_action`, each a real
(simulated) network round-trip, before the LLM can compose a final answer.
Per the product requirement ("while the process waits for the NBA's
response, have an agent keep chatting with the user"), the webhook
handler (`api/routers/telegram_webhook.py`) runs this lightweight,
tool-less chain concurrently with the main agent: it immediately sends a
Telegram "typing..." action plus one short acknowledgement message, then
the real answer follows when the main agent finishes.

Implemented as a plain LCEL `Runnable` (`prompt | llm | output_parser`) —
deliberately not an agent with tools, so it can never itself call
`get_next_best_action` and race the main agent or double-charge the model
gateway.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from financial_agent.agent.output_parser import FillerReply, filler_reply_parser

_FILLER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the same bank financial assistant on Telegram. The customer "
            "just sent a message and the system is preparing a complete response "
            "(it may take a few seconds). Generate only a short, warm sentence "
            "acknowledging the customer's message, without answering the question "
            "itself and without mentioning internal systems.\n\n"
            "{format_instructions}",
        ),
        ("human", "{input}"),
    ]
).partial(format_instructions=filler_reply_parser.get_format_instructions())


class FillerAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = _FILLER_PROMPT | llm | filler_reply_parser

    async def generate(self, user_message: str) -> FillerReply:
        reply: FillerReply = await self._chain.ainvoke({"input": user_message})
        return reply
