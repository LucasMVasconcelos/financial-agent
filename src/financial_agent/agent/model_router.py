"""Model Router — dispatches each activity to the cheapest model that can do it well.

Not every LLM call in this system carries the same stakes. Deciding which
tool to call, following the compliance guardrails, and composing the final
answer to a customer needs the strongest available model — that's a
regulated-adjacent, high-stakes activity. Acknowledging "I'll get right
back to you!" while the real answer is being prepared, or compressing ten old chat
messages into two sentences, does not; a smaller/cheaper model does those
just as well, faster and for a fraction of the cost.

`ModelRouter` is the single place that encodes this mapping, along **two
independent axes**:

  * `for_activity(ModelActivity)` — routes by *what kind of work* is
    happening (filler reply vs. summarization vs. the main agent turn).
    Fixed per call site, doesn't vary by message content.
  * `for_complexity(QueryComplexity)` — routes the main agent turn itself
    by *how demanding this specific message is* (see
    `query_complexity.py`). A simple factual lookup and a multi-part
    question (or one touching the loan tool) don't deserve the same model.

Call sites never construct a `ChatOpenAI` themselves or hardcode a model
name — they ask the router for the model appropriate to what they're about
to do, so the routing policy can change (or grow new tiers) without
touching the agent, the filler chain, or the summarizer.
"""

from __future__ import annotations

from enum import StrEnum

from langchain_core.language_models.chat_models import BaseChatModel

from financial_agent.agent.llm_factory import build_chat_model
from financial_agent.agent.query_complexity import QueryComplexity
from financial_agent.config import Settings


class ModelActivity(StrEnum):
    """Every LLM-backed activity in this system, mapped to a tier below."""

    AGENT_REASONING = "agent_reasoning"
    """The tool-calling agent: picks tools, interprets results, writes the
    customer-facing answer under compliance guardrails. Reasoning tier."""

    FILLER_REPLY = "filler_reply"
    """One short, tool-less acknowledgement sentence sent while the agent
    works. Utility tier."""

    SUMMARIZATION = "summarization"
    """Compressing older conversation turns into a running summary — a
    mechanical text-compression task. Utility tier."""

    JUDGE = "judge"
    """LLM-as-a-judge grading a Golden Transcript response against a rubric
    (`tests/golden/judge.py`) — genuine qualitative judgment (tone,
    faithfulness to retrieved content, guardrail adherence), not a
    mechanical check. Reasoning tier, same as the agent it's grading."""


_REASONING_TIER_ACTIVITIES = frozenset({ModelActivity.AGENT_REASONING, ModelActivity.JUDGE})


class ModelRouter:
    """Holds one Chat Model per tier and dispatches by `ModelActivity`."""

    def __init__(self, *, reasoning_model: BaseChatModel, utility_model: BaseChatModel) -> None:
        self._reasoning_model = reasoning_model
        self._utility_model = utility_model

    def for_activity(self, activity: ModelActivity) -> BaseChatModel:
        if activity in _REASONING_TIER_ACTIVITIES:
            return self._reasoning_model
        return self._utility_model

    def for_complexity(self, complexity: QueryComplexity) -> BaseChatModel:
        """Route a single agent turn by the complexity of that message.

        Used instead of `for_activity(AGENT_REASONING)` at the one call site
        that has the message text available (the webhook handler) — see
        `query_complexity.classify_query_complexity`.
        """
        if complexity is QueryComplexity.COMPLEX:
            return self._reasoning_model
        return self._utility_model

    @classmethod
    def build(cls, settings: Settings) -> ModelRouter:
        """Construct the router from `Settings.openai_reasoning_model` / `_utility_model`."""
        return cls(
            reasoning_model=build_chat_model(settings, model=settings.openai_reasoning_model),
            utility_model=build_chat_model(settings, model=settings.openai_utility_model),
        )
