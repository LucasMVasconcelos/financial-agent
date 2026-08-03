from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from financial_agent.agent.model_router import ModelActivity, ModelRouter
from financial_agent.agent.query_complexity import QueryComplexity


class TestModelRouter:
    def test_agent_reasoning_uses_the_reasoning_tier(self) -> None:
        reasoning = FakeListChatModel(responses=["r"])
        utility = FakeListChatModel(responses=["u"])
        router = ModelRouter(reasoning_model=reasoning, utility_model=utility)

        assert router.for_activity(ModelActivity.AGENT_REASONING) is reasoning

    def test_filler_and_summarization_use_the_utility_tier(self) -> None:
        reasoning = FakeListChatModel(responses=["r"])
        utility = FakeListChatModel(responses=["u"])
        router = ModelRouter(reasoning_model=reasoning, utility_model=utility)

        assert router.for_activity(ModelActivity.FILLER_REPLY) is utility
        assert router.for_activity(ModelActivity.SUMMARIZATION) is utility

    def test_judge_uses_the_reasoning_tier(self) -> None:
        reasoning = FakeListChatModel(responses=["r"])
        utility = FakeListChatModel(responses=["u"])
        router = ModelRouter(reasoning_model=reasoning, utility_model=utility)

        assert router.for_activity(ModelActivity.JUDGE) is reasoning

    def test_build_wires_settings_model_names(self) -> None:
        from financial_agent.config import Settings

        settings = Settings(
            openai_api_key="test-key",
            openai_reasoning_model="gpt-4o",
            openai_utility_model="gpt-4o-mini",
        )
        router = ModelRouter.build(settings)

        reasoning = router.for_activity(ModelActivity.AGENT_REASONING)
        utility = router.for_activity(ModelActivity.FILLER_REPLY)

        assert reasoning.model_name == "gpt-4o"  # type: ignore[attr-defined]
        assert utility.model_name == "gpt-4o-mini"  # type: ignore[attr-defined]
        assert reasoning is not utility

    def test_for_complexity_routes_complex_to_reasoning_tier(self) -> None:
        reasoning = FakeListChatModel(responses=["r"])
        utility = FakeListChatModel(responses=["u"])
        router = ModelRouter(reasoning_model=reasoning, utility_model=utility)

        assert router.for_complexity(QueryComplexity.COMPLEX) is reasoning
        assert router.for_complexity(QueryComplexity.SIMPLE) is utility
