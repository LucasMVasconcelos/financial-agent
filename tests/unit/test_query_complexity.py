from __future__ import annotations

import pytest

from financial_agent.agent.query_complexity import QueryComplexity, classify_query_complexity


class TestClassifyQueryComplexity:
    @pytest.mark.parametrize(
        "text",
        [
            "how does a cd work?",
            "what's my balance?",
            "hi",
            "what products do you have?",
        ],
    )
    def test_short_factual_questions_are_simple(self, text: str) -> None:
        assert classify_query_complexity(text) is QueryComplexity.SIMPLE

    def test_empty_text_is_simple(self) -> None:
        assert classify_query_complexity("   ") is QueryComplexity.SIMPLE

    def test_long_message_is_complex(self) -> None:
        text = " ".join(["word"] * 30)
        assert classify_query_complexity(text) is QueryComplexity.COMPLEX

    @pytest.mark.parametrize(
        "text",
        [
            "I want to request a loan of 10000 reais",
            "why do you recommend this to me?",
            "compare CD and Tesouro Selic",
            "what's the difference between the two products?",
        ],
    )
    def test_signal_words_are_complex(self, text: str) -> None:
        assert classify_query_complexity(text) is QueryComplexity.COMPLEX

    def test_multiple_question_marks_is_complex(self) -> None:
        text = "what about insurance? what about the card?"
        assert classify_query_complexity(text) is QueryComplexity.COMPLEX

    def test_is_case_insensitive(self) -> None:
        assert classify_query_complexity("LOAN") is QueryComplexity.COMPLEX
