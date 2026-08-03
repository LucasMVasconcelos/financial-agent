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

from financial_agent.agent.conversation_summarizer import ConversationSummarizer
from financial_agent.agent.filler_agent import FillerAgent
from financial_agent.agent.llm_factory import build_embeddings, configure_langsmith_tracing
from financial_agent.agent.loan_graph import build_loan_graph
from financial_agent.agent.model_router import ModelActivity, ModelRouter
from financial_agent.config import Settings
from financial_agent.gateways.knowledge_base_gateway import InMemoryKnowledgeBaseGateway
from financial_agent.gateways.nba_model_gateway import (
    MockNBAModelGateway,
    NBAModelGateway,
    SageMakerNBAModelGateway,
)
from financial_agent.gateways.semantic_memory_gateway import InMemorySemanticMemoryGateway
from financial_agent.gateways.telegram_gateway import TelegramGateway
from financial_agent.repositories.conversation_repository import InMemoryConversationRepository
from financial_agent.repositories.customer_repository import InMemoryCustomerRepository
from financial_agent.repositories.loan_repository import InMemoryLoanRepository
from financial_agent.security.rate_limit import InMemoryRateLimiter, RateLimiter
from financial_agent.services.conversation_service import ConversationService
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.knowledge_base_service import KnowledgeBaseService
from financial_agent.services.loan_service import LoanService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService
from financial_agent.services.semantic_memory_service import SemanticMemoryService


@dataclass
class AppState:
    settings: Settings
    http_client: httpx.AsyncClient
    telegram_gateway: TelegramGateway
    customer_service: CustomerService
    nba_service: NBAService
    products_service: ProductsService
    conversation_service: ConversationService
    knowledge_base_service: KnowledgeBaseService
    loan_service: LoanService
    model_router: ModelRouter
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

    embeddings = build_embeddings(settings)
    knowledge_base_gateway = await InMemoryKnowledgeBaseGateway.build(embeddings)
    semantic_memory_gateway = InMemorySemanticMemoryGateway(embeddings)
    semantic_memory_service = SemanticMemoryService(semantic_memory_gateway)

    model_router = ModelRouter.build(settings)
    utility_model = model_router.for_activity(ModelActivity.SUMMARIZATION)

    conversation_service = ConversationService(
        conversation_repository,
        ConversationSummarizer(utility_model),
        semantic_memory_service,
    )

    customer_service = CustomerService(customer_repository)
    loan_repository = InMemoryLoanRepository()
    loan_graph = build_loan_graph(
        customer_service=customer_service,
        approval_threshold=settings.loan_human_approval_threshold,
    )
    loan_service = LoanService(
        graph=loan_graph,
        loan_repository=loan_repository,
        telegram_gateway=telegram_gateway,
    )

    return AppState(
        settings=settings,
        http_client=http_client,
        telegram_gateway=telegram_gateway,
        customer_service=customer_service,
        nba_service=NBAService(customer_repository, nba_model_gateway),
        products_service=ProductsService(customer_repository),
        conversation_service=conversation_service,
        knowledge_base_service=KnowledgeBaseService(knowledge_base_gateway),
        loan_service=loan_service,
        model_router=model_router,
        filler_agent=FillerAgent(model_router.for_activity(ModelActivity.FILLER_REPLY)),
        rate_limiter=InMemoryRateLimiter(
            max_requests=settings.rate_limit_max_requests,
            window_seconds=settings.rate_limit_window_seconds,
        ),
    )


async def shutdown_app_state(state: AppState) -> None:
    # telegram_gateway wraps the same http_client instance, closing it once suffices.
    await state.telegram_gateway.aclose()
