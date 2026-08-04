"""Unit tests for `ConversationSummarizer`.

Uses `FakeListChatModel` (network-free, returns canned responses in order)
so this test never depends on a real OpenAI call — it verifies the chain's
wiring (prompt -> llm -> StrOutputParser, transcript formatting), not
summary quality, which needs a real model and is out of scope here.
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from financial_agent.agent.conversation_summarizer import ConversationSummarizer
from financial_agent.domain.models.conversation import ConversationMessage, MessageRole


def _message(content: str) -> ConversationMessage:
    return ConversationMessage(role=MessageRole.USER, content=content, timestamp=datetime.now(UTC))


class TestConversationSummarizer:
    async def test_returns_the_model_response_stripped(self) -> None:
        llm = FakeListChatModel(responses=["  Customer asked about CDs.  "])
        summarizer = ConversationSummarizer(llm)

        result = await summarizer.summarize(
            existing_summary=None, new_turns=[_message("how does a CD work?")]
        )

        assert result == "Customer asked about CDs."

    async def test_works_with_an_existing_summary_and_multiple_turns(self) -> None:
        llm = FakeListChatModel(responses=["Updated summary."])
        summarizer = ConversationSummarizer(llm)

        result = await summarizer.summarize(
            existing_summary="Customer already asked about Tesouro Selic.",
            new_turns=[_message("what about CDs?"), _message("thanks!")],
        )

        assert result == "Updated summary."

    async def test_empty_new_turns_still_produces_a_summary(self) -> None:
        llm = FakeListChatModel(responses=["Nothing new."])
        summarizer = ConversationSummarizer(llm)

        result = await summarizer.summarize(existing_summary="something", new_turns=[])

        assert result == "Nothing new."
