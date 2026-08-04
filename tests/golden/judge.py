"""LLM-as-a-judge — the qualitative evaluation layer for Golden Transcripts.

`ResponseAssertions` (substring matching) and `expected_tool_calls`
(mechanical) cover what can be checked deterministically. Some things
can't: whether the tone matches the persona the system prompt asks for,
whether an explanation actually makes sense, whether the answer is
*faithful* to what a tool returned rather than just happening to mention
the right keyword. That's what `rubric` + this module are for — a second,
separate LLM call that grades the response against one free-text criterion.

Deliberately narrow: one rubric in, one pass/fail + reasoning out. No
multi-turn judge conversation, no numeric scoring scale — a golden
transcript is a regression gate, not a leaderboard, and a binary verdict
with a reason is both easier to act on and easier to keep honest than a
"7/10" that nobody can explain a month later.

Uses the Model Router's `JUDGE` activity (reasoning tier, same rationale as
the agent it's grading — judging tone and faithfulness is real judgment,
not mechanical work) — see `agent/model_router.py`.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You evaluate whether a financial assistant's response satisfies ONE specific "
            "criterion (the rubric). Be strict and objective: the goal is to catch behavior "
            "regressions, not to praise the response. Evaluate only what the rubric asks "
            "for, nothing beyond that — don't penalize for style or length if the rubric "
            "doesn't mention it. In case of genuine doubt, fail it and explain the doubt.",
        ),
        (
            "human",
            "Scenario: {description}\n\n"
            "Rubric (criterion to evaluate): {rubric}\n\n"
            "Assistant's response to be evaluated:\n{response}\n\n"
            "Does the response satisfy the rubric?",
        ),
    ]
)


class JudgeVerdict(BaseModel):
    passed: bool = Field(description="Whether the response satisfies the rubric.")
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


class GoldenTranscriptJudge:
    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = _JUDGE_PROMPT | llm.with_structured_output(JudgeVerdict)

    async def evaluate(self, *, description: str, rubric: str, response: str) -> JudgeVerdict:
        verdict = await self._chain.ainvoke(
            {"description": description, "rubric": rubric, "response": response}
        )
        assert isinstance(verdict, JudgeVerdict)  # narrows with_structured_output's Any return
        return verdict
