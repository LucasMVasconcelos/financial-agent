"""Application configuration.

All configuration is sourced from environment variables (12-factor style),
loaded once into a cached `Settings` singleton. Nothing else in the codebase
should call `os.environ` directly — this is the single seam that maps the
outside world (env vars, `.env` file) into typed, validated config.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"

    # Telegram
    telegram_bot_token: str = "change-me"
    telegram_webhook_secret: str = "change-me"

    # Internal service auth (non-Telegram callers)
    service_api_key: str = "change-me"

    # LLM provider
    openai_api_key: str = "change-me"
    openai_temperature: float = 0.2
    openai_embedding_model: str = "text-embedding-3-small"

    # Model Router: which concrete model backs each tier. The reasoning tier
    # handles the tool-calling agent (needs strong instruction-following and
    # judgment); the utility tier handles cheap/mechanical activities (filler
    # replies, conversation summarization) where a smaller model is enough
    # and latency/cost matter more. See `agent/model_router.py`.
    openai_reasoning_model: str = "gpt-4o"
    openai_utility_model: str = "gpt-4o-mini"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "financial-agent"

    # Rate limiting
    rate_limit_max_requests: int = 20
    rate_limit_window_seconds: int = 60

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    use_redis: bool = False

    # NBA model gateway
    nba_model_provider: Literal["mock", "sagemaker"] = "mock"
    aws_region: str = "us-east-1"
    sagemaker_endpoint_name: str = ""

    # Loan origination (agent/loan_graph.py)
    # Applications above this amount (BRL) pause for human approval instead
    # of auto-approving.
    loan_human_approval_threshold: float = 50_000.0

    # Filler reply (api/routers/telegram_webhook.py). The filler message is
    # only sent if the main agent hasn't replied within this many seconds —
    # sending it unconditionally means a fast answer arrives right behind a
    # "hold on" message the customer never needed, which reads as spam
    # rather than as the latency-hiding UX it's meant to be.
    filler_delay_seconds: float = 2.5


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance (env is read once)."""
    return Settings()
