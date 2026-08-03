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
        llm = FakeListChatModel(responses=["  Cliente perguntou sobre CDB.  "])
        summarizer = ConversationSummarizer(llm)

        result = await summarizer.summarize(
            existing_summary=None, new_turns=[_message("como funciona o cdb?")]
        )

        assert result == "Cliente perguntou sobre CDB."

    async def test_works_with_an_existing_summary_and_multiple_turns(self) -> None:
        llm = FakeListChatModel(responses=["Resumo atualizado."])
        summarizer = ConversationSummarizer(llm)

        result = await summarizer.summarize(
            existing_summary="Cliente já perguntou sobre Tesouro Selic.",
            new_turns=[_message("e sobre CDB?"), _message("obrigado!")],
        )

        assert result == "Resumo atualizado."

    async def test_empty_new_turns_still_produces_a_summary(self) -> None:
        llm = FakeListChatModel(responses=["Sem novidades."])
        summarizer = ConversationSummarizer(llm)

        result = await summarizer.summarize(existing_summary="algo", new_turns=[])

        assert result == "Sem novidades."
