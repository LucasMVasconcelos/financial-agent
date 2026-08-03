"""Unit tests for `ConversationService`: assembling turn memory (raw window +
summary + long-term recall) and the compaction policy that feeds the
summary and long-term memory from the raw window over time.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from financial_agent.agent.conversation_summarizer import ConversationSummarizer
from financial_agent.domain.models.conversation import MessageRole
from financial_agent.repositories.conversation_repository import InMemoryConversationRepository
from financial_agent.services.conversation_service import (
    _COMPACT_CHUNK,
    _MAX_RAW_MESSAGES,
    ConversationService,
)

USER_ID = 123


class _SpySemanticMemoryService:
    def __init__(self, *, recall_results: list[str] | None = None) -> None:
        self.remembered: list[tuple[int, str]] = []
        self._recall_results = recall_results or []

    async def remember(self, user_id: int, text: str) -> None:
        self.remembered.append((user_id, text))

    async def recall(self, user_id: int, query: str, *, top_k: int = 3) -> list[str]:
        return self._recall_results


class _FailingSummarizer:
    async def summarize(self, *, existing_summary: str | None, new_turns: list[object]) -> str:
        raise RuntimeError("model backend unavailable")


def _build_service(
    *, summary_response: str = "resumo-fake", recall_results: list[str] | None = None
) -> tuple[ConversationService, _SpySemanticMemoryService]:
    repository = InMemoryConversationRepository()
    summarizer = ConversationSummarizer(FakeListChatModel(responses=[summary_response]))
    semantic_memory = _SpySemanticMemoryService(recall_results=recall_results)
    return ConversationService(repository, summarizer, semantic_memory), semantic_memory


class TestConversationServiceHistory:
    async def test_get_history_overlays_long_term_recall(self) -> None:
        service, _ = _build_service(recall_results=["cliente já perguntou sobre CDB"])

        await service.record_turn(USER_ID, role=MessageRole.USER, content="oi")

        history = await service.get_history(USER_ID, current_message="tudo bem?")

        assert [m.content for m in history.messages] == ["oi"]
        assert history.long_term_memories == ["cliente já perguntou sobre CDB"]

    async def test_get_history_for_unknown_user_is_empty(self) -> None:
        service, _ = _build_service()

        history = await service.get_history(999_999, current_message="oi")

        assert history.messages == []
        assert history.summary is None
        assert history.long_term_memories == []


class TestConversationServiceCompaction:
    async def test_below_threshold_never_compacts(self) -> None:
        service, spy = _build_service()

        for i in range(5):
            await service.record_turn(USER_ID, role=MessageRole.USER, content=f"mensagem {i}")

        history = await service.get_history(USER_ID, current_message="oi")

        assert len(history.messages) == 5
        assert history.summary is None
        assert spy.remembered == []

    async def test_crossing_threshold_compacts_oldest_chunk(self) -> None:
        service, spy = _build_service(summary_response="Resumo compactado.")

        total_messages = _MAX_RAW_MESSAGES + 1
        for i in range(total_messages):
            await service.record_turn(USER_ID, role=MessageRole.USER, content=f"mensagem {i}")

        history = await service.get_history(USER_ID, current_message="oi")

        assert len(history.messages) == total_messages - _COMPACT_CHUNK
        assert history.summary == "Resumo compactado."
        # the oldest messages are the ones summarized away
        assert history.messages[0].content == f"mensagem {_COMPACT_CHUNK}"

    async def test_compaction_files_the_summary_as_a_long_term_memory(self) -> None:
        service, spy = _build_service(summary_response="Resumo compactado.")

        for i in range(_MAX_RAW_MESSAGES + 1):
            await service.record_turn(USER_ID, role=MessageRole.USER, content=f"mensagem {i}")

        assert spy.remembered == [(USER_ID, "Resumo compactado.")]

    async def test_summarizer_failure_never_raises_and_skips_compaction(self) -> None:
        repository = InMemoryConversationRepository()
        semantic_memory = _SpySemanticMemoryService()
        service = ConversationService(repository, _FailingSummarizer(), semantic_memory)  # type: ignore[arg-type]

        for i in range(_MAX_RAW_MESSAGES + 1):
            await service.record_turn(USER_ID, role=MessageRole.USER, content=f"mensagem {i}")

        history = await service.get_history(USER_ID, current_message="oi")

        assert len(history.messages) == _MAX_RAW_MESSAGES + 1  # nothing was trimmed
        assert history.summary is None
        assert semantic_memory.remembered == []
