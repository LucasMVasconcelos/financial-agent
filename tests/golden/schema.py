"""Golden Transcripts — typed schema for the project's LLM-behavior regression suite.

A "golden transcript" is a curated conversation scenario with *behavioral*
expectations about the agent's response — which tools must (or must never)
be called, and what the final answer must/must not contain. This is a
different kind of test than everything else in `tests/`: unit/contract/api
tests exercise code paths deterministically and never touch a real LLM;
golden transcripts exercise the actual agent (a real `ChatOpenAI` call) and
therefore accept non-determinism in *wording* while still asserting hard
constraints on *behavior* — which is exactly the split this schema encodes:

  * `expected_tool_calls` — deterministic, mechanical checks (a tool was or
    wasn't invoked, with what arguments). These can be asserted exactly.
  * `response_assertions` — soft, substring-based checks on the final
    answer. Deliberately not exact-text matching: the same prompt can (and
    should be expected to) produce different wording across runs, even at
    low temperature. What must stay constant is *content*, not phrasing —
    e.g. a loan pending human approval must never say "approved", no matter
    how the sentence around it is worded.
  * `rubric` — an optional, free-text criterion graded by a separate LLM
    call (`tests/golden/judge.py`), for the qualities substring matching
    structurally cannot check: tone, whether an explanation is coherent,
    whether the answer is actually faithful to what a tool returned rather
    than just mentioning the right keyword. Additive, not a replacement —
    `expected_tool_calls` and `response_assertions` stay the fast,
    deterministic, free first line of defense; the judge is a slower,
    non-deterministic second opinion reserved for what they can't see.

See `tests/golden/transcripts.yaml` for the actual scenarios and
`tests/golden/run_golden_transcripts.py` for how they'd be executed against
a live agent (requires a real `OPENAI_API_KEY` — not part of `pytest`, see
that module's docstring for why).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class GoldenTranscriptCategory(StrEnum):
    NBA = "nba"
    PROFILE = "profile"
    PRODUCTS = "products"
    RAG = "rag"
    LOAN = "loan"
    GUARDRAIL = "guardrail"
    SECURITY = "security"
    ERROR_HANDLING = "error_handling"
    MULTI_TOOL = "multi_tool"


class ExpectedToolCall(BaseModel):
    """One assertion about whether/how a tool should be invoked during the run."""

    tool_name: str
    required: bool = Field(
        default=False,
        description="If true, the tool must be called at least once. Defaults to False so "
        "`forbidden: true` entries don't also need to spell out `required: false`.",
    )
    forbidden: bool = Field(
        default=False,
        description="If true, the tool must never be called — for negative/security checks.",
    )
    input_contains: dict[str, object] | None = Field(
        default=None,
        description=(
            "Partial match: if set, at least one call to this tool must have been made "
            "with arguments containing these key/value pairs."
        ),
    )

    @model_validator(mode="after")
    def _not_both_required_and_forbidden(self) -> ExpectedToolCall:
        if self.required and self.forbidden:
            raise ValueError("A tool assertion cannot be both required and forbidden.")
        return self

    @model_validator(mode="after")
    def _asserts_something(self) -> ExpectedToolCall:
        if not self.required and not self.forbidden and self.input_contains is None:
            raise ValueError(
                f"'{self.tool_name}' assertion checks nothing — set required, forbidden, "
                "or input_contains (likely a missing `required: true`)."
            )
        return self


class ResponseAssertions(BaseModel):
    """Substring-based behavioral checks on the agent's final answer.

    Case-insensitive. Empty lists mean "no assertion" — some scenarios (pure
    refusals, tone-only checks) are better verified by a human reading the
    transcript than by brittle substring matching; see each scenario's
    `notes` for those cases.
    """

    must_mention_any: list[str] = Field(
        default_factory=list,
        description="At least one of these substrings must appear (empty = no check).",
    )
    must_not_contain: list[str] = Field(
        default_factory=list,
        description="None of these substrings may ever appear — guardrail violations, "
        "raw error codes, leaked data from another customer, etc.",
    )


class GoldenTranscript(BaseModel):
    id: str
    description: str
    category: GoldenTranscriptCategory
    user_id: int = Field(description="Authenticated user_id this scenario runs as.")
    turns: list[str] = Field(min_length=1, description="User messages, in order.")
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    response_assertions: ResponseAssertions = Field(default_factory=ResponseAssertions)
    rubric: str | None = Field(
        default=None,
        description="Free-text criterion for the LLM-as-a-judge pass (tests/golden/judge.py). "
        "None means no judge call is made for this transcript.",
    )
    notes: str | None = None


class GoldenTranscriptSuite(BaseModel):
    version: str
    transcripts: list[GoldenTranscript]


def load_golden_transcripts(path: Path) -> GoldenTranscriptSuite:
    """Parse and validate the YAML file into a `GoldenTranscriptSuite`."""
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return GoldenTranscriptSuite.model_validate(raw)
