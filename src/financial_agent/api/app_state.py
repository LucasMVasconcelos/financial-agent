"""Process-wide singletons, built once at startup and injected via `api/deps.py`.

This is the Composition Root: the one place that knows about every concrete
implementation (in-memory repositories, the mock/SageMaker NBA gateway,
httpx clients, ...) and wires them behind the Protocols the rest of the
codebase depends on. Swapping an implementation (e.g. in-memory repository
-> Postgres repository) means changing `build_app_state` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from langchain_core.language_models.chat_models import BaseChatModel

from financial_agent.agent.filler_agent import FillerAgent
from financial_agent.agent.llm_factory import build_chat_model, configure_langsmith_tracing
from financial_agent.config import Settings
from financial_agent.gateways.nba_model_gateway import (
    MockNBAModelGateway,
    NBAModelGateway,
    SageMakerNBAModelGateway,
)
from financial_agent.gateways.telegram_gateway import TelegramGateway
from financial_agent.repositories.conversation_repository import InMemoryConversationRepository
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.security.rate_limit import InMemoryRateLimiter, RateLimiter
from financial_agent.services.conversation_service import ConversationService
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService


@dataclass
class AppState:
    settings: Settings
    http_client: httpx.AsyncClient
    telegram_gateway: TelegramGateway
    customer_service: CustomerService
    nba_service: NBAService
    products_service: ProductsService
    conversation_service: ConversationService
    llm: BaseChatModel
    filler_agent: FillerAgent
    rate_limiter: RateLimiter


def _build_nba_model_gateway(settings: Settings) -> NBAModelGateway:
    if settings.nba_model_provider == "sagemaker":
        return SageMakerNBAModelGateway(
            endpoint_name=settings.sagemaker_endpoint_name,
            region_name=settings.aws_region,
        )
    return MockNBAModelGateway()


async def build_app_state(settings: Settings) -> AppState:
    configure_langsmith_tracing(settings)

    http_client = httpx.AsyncClient(timeout=10.0)
    telegram_gateway = TelegramGateway(
        bot_token=settings.telegram_bot_token, http_client=http_client
    )

    customer_repository = InMemoryCustomerRepository()
    conversation_repository = InMemoryConversationRepository()
    nba_model_gateway = _build_nba_model_gateway(settings)

    llm = build_chat_model(settings)

    return AppState(
        settings=settings,
        http_client=http_client,
        telegram_gateway=telegram_gateway,
        customer_service=CustomerService(customer_repository),
        nba_service=NBAService(customer_repository, nba_model_gateway),
        products_service=ProductsService(customer_repository),
        conversation_service=ConversationService(conversation_repository),
        llm=llm,
        filler_agent=FillerAgent(llm),
        rate_limiter=InMemoryRateLimiter(
            max_requests=settings.rate_limit_max_requests,
            window_seconds=settings.rate_limit_window_seconds,
        ),
    )


async def shutdown_app_state(state: AppState) -> None:
    # telegram_gateway wraps the same http_client instance, closing it once suffices.
    await state.telegram_gateway.aclose()
