from __future__ import annotations

import pytest

from financial_agent.agent.query_complexity import QueryComplexity, classify_query_complexity


class TestClassifyQueryComplexity:
    @pytest.mark.parametrize(
        "text",
        [
            "como funciona o cdb?",
            "qual meu saldo?",
            "oi",
            "quais produtos vocês têm?",
        ],
    )
    def test_short_factual_questions_are_simple(self, text: str) -> None:
        assert classify_query_complexity(text) is QueryComplexity.SIMPLE

    def test_empty_text_is_simple(self) -> None:
        assert classify_query_complexity("   ") is QueryComplexity.SIMPLE

    def test_long_message_is_complex(self) -> None:
        text = " ".join(["palavra"] * 30)
        assert classify_query_complexity(text) is QueryComplexity.COMPLEX

    @pytest.mark.parametrize(
        "text",
        [
            "quero solicitar um empréstimo de 10000 reais",
            "por que vocês recomendam isso pra mim?",
            "compare CDB e Tesouro Selic",
            "qual a diferença entre os dois produtos?",
        ],
    )
    def test_signal_words_are_complex(self, text: str) -> None:
        assert classify_query_complexity(text) is QueryComplexity.COMPLEX

    def test_multiple_question_marks_is_complex(self) -> None:
        text = "e sobre seguros? e sobre cartão?"
        assert classify_query_complexity(text) is QueryComplexity.COMPLEX

    def test_is_case_insensitive(self) -> None:
        assert classify_query_complexity("EMPRÉSTIMO") is QueryComplexity.COMPLEX
