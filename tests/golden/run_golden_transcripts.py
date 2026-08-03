"""Executes the golden transcripts against the *real* agent — real ChatOpenAI calls.

Deliberately **not** a `test_*.py` file: pytest never collects this, and it
never runs in CI. Every other test in this repository is network-free by
design (fakes, `DeterministicFakeEmbedding`, monkeypatched LLM calls) so the
suite stays fast, deterministic, and free to run on every commit — golden
transcripts are the opposite on purpose: they exist to catch prompt/model
regressions that only show up with a real model in the loop, which costs
money and is never fully deterministic. Run this by hand (or from a
separate, manually-triggered CI job) when you change the system prompt, a
Tool description, or the model powering either tier.

Every transcript gets the deterministic checks (`expected_tool_calls`,
`response_assertions`). Transcripts with a `rubric` also get a second,
qualitative pass from `tests.golden.judge.GoldenTranscriptJudge` — an LLM
call grading the response against that one free-text criterion, for the
things substring matching can't see (tone, faithfulness to retrieved
content). That's one extra real LLM call per rubric — another reason this
stays out of the on-commit suite.

Usage:
    OPENAI_API_KEY=sk-... poetry run python tests/golden/run_golden_transcripts.py
    OPENAI_API_KEY=sk-... poetry run python tests/golden/run_golden_transcripts.py \\
        --id loan_large_amount_requires_human_approval

Exit code is non-zero if any transcript fails — wire this into a scheduled
CI job (not the on-commit one) if/when this project gets one.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tests.golden.judge import GoldenTranscriptJudge  # noqa: E402
from tests.golden.schema import (  # noqa: E402
    ExpectedToolCall,
    GoldenTranscript,
    ResponseAssertions,
    load_golden_transcripts,
)

from financial_agent.agent.main_graph import build_main_graph, run_main_graph_turn  # noqa: E402
from financial_agent.agent.model_router import ModelActivity  # noqa: E402
from financial_agent.agent.query_complexity import classify_query_complexity  # noqa: E402
from financial_agent.api.app_state import build_app_state, shutdown_app_state  # noqa: E402
from financial_agent.config import get_settings  # noqa: E402
from financial_agent.domain.models.conversation import ConversationHistory  # noqa: E402

_TRANSCRIPTS_PATH = Path(__file__).parent / "transcripts.yaml"


class _ToolCallRecorder(AsyncCallbackHandler):
    """Records every tool invocation (name + parsed args) during one agent turn."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "<unknown>")
        self.calls.append((name, inputs or {}))


def _check_tool_calls(
    expected: list[ExpectedToolCall], actual: list[tuple[str, dict[str, Any]]]
) -> list[str]:
    failures = []
    called_names = {name for name, _ in actual}
    for assertion in expected:
        if assertion.forbidden and assertion.tool_name in called_names:
            failures.append(f"'{assertion.tool_name}' was called but is forbidden")
        if assertion.required and assertion.tool_name not in called_names:
            failures.append(f"'{assertion.tool_name}' was required but never called")
        if assertion.input_contains:
            matches = [
                args
                for name, args in actual
                if name == assertion.tool_name and assertion.input_contains.items() <= args.items()
            ]
            if not matches:
                failures.append(
                    f"'{assertion.tool_name}' was never called with {assertion.input_contains}"
                )
    return failures


def _check_response(assertions: ResponseAssertions, response: str) -> list[str]:
    failures = []
    lowered = response.lower()
    if assertions.must_mention_any and not any(
        s.lower() in lowered for s in assertions.must_mention_any
    ):
        failures.append(f"response mentions none of {assertions.must_mention_any}")
    for forbidden in assertions.must_not_contain:
        if forbidden.lower() in lowered:
            failures.append(f"response contains forbidden text: '{forbidden}'")
    return failures


async def _run_one(transcript: GoldenTranscript) -> list[str]:
    settings = get_settings()
    app_state = await build_app_state(settings)
    recorder = _ToolCallRecorder()
    try:
        history = ConversationHistory(user_id=transcript.user_id)
        final_response = ""
        for turn_index, turn in enumerate(transcript.turns):
            llm = app_state.model_router.for_complexity(classify_query_complexity(turn))
            graph = build_main_graph(
                user_id=transcript.user_id,
                llm=llm,
                customer_service=app_state.customer_service,
                nba_service=app_state.nba_service,
                products_service=app_state.products_service,
                knowledge_base_service=app_state.knowledge_base_service,
                loan_service=app_state.loan_service,
                checkpointer=app_state.checkpointer,
            )
            final_response = await run_main_graph_turn(
                graph=graph.with_config(callbacks=[recorder]),
                thread_id=f"golden:{transcript.id}:{turn_index}",
                user_id=transcript.user_id,
                user_message=turn,
                history=history,
            )
            history = await app_state.conversation_service.get_history(
                transcript.user_id, current_message=turn
            )

        failures = _check_tool_calls(transcript.expected_tool_calls, recorder.calls)
        failures += _check_response(transcript.response_assertions, final_response)

        if transcript.rubric:
            judge = GoldenTranscriptJudge(app_state.model_router.for_activity(ModelActivity.JUDGE))
            verdict = await judge.evaluate(
                description=transcript.description,
                rubric=transcript.rubric,
                response=final_response,
            )
            if not verdict.passed:
                failures.append(f"[judge] {verdict.reasoning}")

        return failures
    finally:
        await shutdown_app_state(app_state)


async def main(transcript_id: str | None) -> int:
    suite = load_golden_transcripts(_TRANSCRIPTS_PATH)
    transcripts = suite.transcripts
    if transcript_id:
        transcripts = [t for t in suite.transcripts if t.id == transcript_id]
    if not transcripts:
        print(f"No transcript matches id={transcript_id!r}")
        return 1

    exit_code = 0
    for transcript in transcripts:
        failures = await _run_one(transcript)
        status = "PASS" if not failures else "FAIL"
        print(f"[{status}] {transcript.id} — {transcript.description.strip()}")
        for failure in failures:
            print(f"         - {failure}")
        if failures:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id", dest="transcript_id", default=None, help="Run only this transcript id"
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.transcript_id)))
