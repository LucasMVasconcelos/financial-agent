"""Query complexity classification — the second axis of the Model Router.

`ModelActivity` (see `model_router.py`) routes by *what kind of work* is
happening (filler reply vs. summarization vs. the main agent turn). This
module adds a second, independent axis for the main agent turn itself:
*how demanding is this specific message*, so a simple "how does CDB work"
question doesn't pay for the reasoning-tier model that a multi-part
question — or a loan request touching the project's one mutating tool —
genuinely needs.

Deliberately a cheap, deterministic heuristic, not an LLM call: spending a
model call just to decide which model to use would undermine the entire
point (saving cost/latency on the easy case). The trade-off is a heuristic
that gives correctness in the common cases, not universal accuracy — that
is judged acceptable here because misclassifying "simple" only costs a
worse-than-ideal model on one turn, never a wrong or unsafe answer (the
reasoning-tier guardrails and tool contracts are unaffected either way).
"""

from __future__ import annotations

from enum import StrEnum

_COMPLEX_WORD_THRESHOLD = 22

# Any of these appearing (case-insensitive) pushes a message to COMPLEX
# regardless of length: financial-movement/loan requests always deserve the
# reasoning tier (the tool involved is the project's one consequential
# action), and comparison/justification requests genuinely need stronger
# reasoning than a single factual lookup.
_COMPLEX_SIGNAL_WORDS = (
    "loan",
    "why",
    "compare",
    "comparison",
    "difference",
    "best for me",
    "recommend",
)


class QueryComplexity(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"


def classify_query_complexity(text: str) -> QueryComplexity:
    """Classify a single user message as SIMPLE or COMPLEX.

    COMPLEX when any of:
      * the message is long (more than `_COMPLEX_WORD_THRESHOLD` words) —
        a proxy for multi-part or elaborate requests;
      * it contains a loan/financial-movement or comparison/justification
        signal word;
      * it has more than one question mark — a proxy for multiple distinct
        asks in one message.

    Otherwise SIMPLE.
    """
    normalized = text.strip().lower()
    if not normalized:
        return QueryComplexity.SIMPLE

    word_count = len(normalized.split())
    if word_count > _COMPLEX_WORD_THRESHOLD:
        return QueryComplexity.COMPLEX

    if any(signal in normalized for signal in _COMPLEX_SIGNAL_WORDS):
        return QueryComplexity.COMPLEX

    if normalized.count("?") > 1:
        return QueryComplexity.COMPLEX

    return QueryComplexity.SIMPLE
