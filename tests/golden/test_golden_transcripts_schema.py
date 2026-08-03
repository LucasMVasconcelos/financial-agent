"""Schema-only validation of the golden transcripts file.

Deliberately does **not** call an LLM — this runs in the normal, network-free
pytest suite on every commit, catching authoring mistakes (bad YAML, unknown
tool names, duplicate ids) immediately. Actually *executing* the scenarios
against a live agent is a separate, manual step — see
`run_golden_transcripts.py`.
"""

from __future__ import annotations

from pathlib import Path

from tests.golden.judge import JudgeVerdict
from tests.golden.schema import (
    GoldenTranscriptCategory,
    GoldenTranscriptSuite,
    load_golden_transcripts,
)

from financial_agent.agent.tools.get_customer_profile import GET_CUSTOMER_PROFILE_TOOL_NAME
from financial_agent.agent.tools.get_next_best_action import GET_NEXT_BEST_ACTION_TOOL_NAME
from financial_agent.agent.tools.get_products import GET_PRODUCTS_TOOL_NAME
from financial_agent.agent.tools.request_loan import REQUEST_LOAN_TOOL_NAME
from financial_agent.agent.tools.search_knowledge_base import SEARCH_KNOWLEDGE_BASE_TOOL_NAME

_TRANSCRIPTS_PATH = Path(__file__).parent / "transcripts.yaml"

_KNOWN_TOOL_NAMES = {
    GET_CUSTOMER_PROFILE_TOOL_NAME,
    GET_NEXT_BEST_ACTION_TOOL_NAME,
    GET_PRODUCTS_TOOL_NAME,
    SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
    REQUEST_LOAN_TOOL_NAME,
}


def _load() -> GoldenTranscriptSuite:
    return load_golden_transcripts(_TRANSCRIPTS_PATH)


class TestGoldenTranscriptsSchema:
    def test_file_parses_and_validates(self) -> None:
        suite = _load()
        assert suite.version
        assert len(suite.transcripts) > 0

    def test_ids_are_unique(self) -> None:
        suite = _load()
        ids = [t.id for t in suite.transcripts]
        assert len(ids) == len(set(ids))

    def test_every_transcript_has_at_least_one_turn(self) -> None:
        suite = _load()
        for transcript in suite.transcripts:
            assert len(transcript.turns) >= 1, transcript.id

    def test_referenced_tool_names_exist_in_the_codebase(self) -> None:
        """Catches drift: a tool gets renamed and the YAML silently references a ghost."""
        suite = _load()
        for transcript in suite.transcripts:
            for expected_call in transcript.expected_tool_calls:
                assert expected_call.tool_name in _KNOWN_TOOL_NAMES, (
                    f"{transcript.id}: unknown tool '{expected_call.tool_name}'"
                )

    def test_every_category_is_represented(self) -> None:
        """A light coverage guard: every category the schema defines should have
        at least one example, otherwise the enum value is dead weight."""
        suite = _load()
        covered = {t.category for t in suite.transcripts}
        assert covered == set(GoldenTranscriptCategory)

    def test_user_ids_referenced_are_the_documented_seeded_set(self) -> None:
        """123/456/1140762405 are seeded; 999999 is the deliberate NOT_FOUND case."""
        suite = _load()
        known_user_ids = {123, 456, 1_140_762_405, 999_999}
        for transcript in suite.transcripts:
            assert transcript.user_id in known_user_ids, transcript.id

    def test_rubrics_are_non_empty_when_present(self) -> None:
        suite = _load()
        for transcript in suite.transcripts:
            if transcript.rubric is not None:
                assert transcript.rubric.strip(), transcript.id

    def test_at_least_one_scenario_has_a_rubric(self) -> None:
        """A light guard against the LLM-as-judge dimension silently going unused."""
        suite = _load()
        assert any(t.rubric for t in suite.transcripts)


class TestJudgeVerdictSchema:
    """`JudgeVerdict` itself needs no LLM to validate — it's a plain pydantic model."""

    def test_accepts_a_passing_verdict(self) -> None:
        verdict = JudgeVerdict(passed=True, reasoning="Atende ao critério.")
        assert verdict.passed is True

    def test_accepts_a_failing_verdict(self) -> None:
        verdict = JudgeVerdict(passed=False, reasoning="Não atende ao critério porque X.")
        assert verdict.passed is False
