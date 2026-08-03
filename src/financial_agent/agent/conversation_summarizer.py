"""Conversation summarizer — folds old turns into a running summary.

A plain, tool-less LCEL chain (`prompt | llm | StrOutputParser`), the same
shape as `agent/filler_agent.py`. Deliberately built on the Model Router's
utility tier (see `agent/model_router.py`): summarization is mechanical
text compression, not judgment — it doesn't need the reasoning tier's cost
or latency.

Called by `ConversationService._maybe_compact` once the raw message window
grows past its cap; never called directly by a route handler.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from financial_agent.domain.models.conversation import ConversationMessage

_SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You summarize conversations between a customer and a financial "
            "assistant. Produce a short summary (5 sentences maximum) in English, "
            "preserving relevant facts about the customer and what has already been "
            "discussed or recommended — products mentioned, recommendations given, "
            "questions already answered. Don't make up anything that isn't in the "
            "text below. Reply with only the summary, no introductions.",
        ),
        (
            "human",
            "Existing summary (may be empty):\n{existing_summary}\n\n"
            "New messages to incorporate:\n{new_turns}\n\n"
            "Write the updated summary, combining what already existed with the new "
            "messages.",
        ),
    ]
)


class ConversationSummarizer:
    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = _SUMMARIZER_PROMPT | llm | StrOutputParser()

    async def summarize(
        self, *, existing_summary: str | None, new_turns: list[ConversationMessage]
    ) -> str:
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in new_turns)
        result: str = await self._chain.ainvoke(
            {
                "existing_summary": existing_summary or "(none yet)",
                "new_turns": transcript,
            }
        )
        return result.strip()
