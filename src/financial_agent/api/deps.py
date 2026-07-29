"""FastAPI dependency providers.

Every provider reads from `request.app.state.container` (an `AppState`
built once at startup — see `api/app_state.py`) rather than constructing
anything itself. This keeps route handlers and the agent layer free of
`import fastapi`, which is what lets the same services/agent code be reused
verbatim from the AWS Lambda handler in `infra/aws/lambda_handler.py`.
"""

from __future__ import annotations

from fastapi import Request
from langchain_core.language_models.chat_models import BaseChatModel

from financial_agent.agent.filler_agent import FillerAgent
from financial_agent.api.app_state import AppState
from financial_agent.config import Settings
from financial_agent.gateways.telegram_gateway import TelegramGateway
from financial_agent.security.rate_limit import RateLimiter
from financial_agent.services.conversation_service import ConversationService
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService


def get_app_state(request: Request) -> AppState:
    state: AppState = request.app.state.container
    return state


def get_settings_dep(request: Request) -> Settings:
    return get_app_state(request).settings


def get_telegram_gateway(request: Request) -> TelegramGateway:
    return get_app_state(request).telegram_gateway


def get_customer_service(request: Request) -> CustomerService:
    return get_app_state(request).customer_service


def get_nba_service(request: Request) -> NBAService:
    return get_app_state(request).nba_service


def get_products_service(request: Request) -> ProductsService:
    return get_app_state(request).products_service


def get_conversation_service(request: Request) -> ConversationService:
    return get_app_state(request).conversation_service


def get_llm(request: Request) -> BaseChatModel:
    return get_app_state(request).llm


def get_filler_agent(request: Request) -> FillerAgent:
    return get_app_state(request).filler_agent


def get_rate_limiter(request: Request) -> RateLimiter:
    return get_app_state(request).rate_limiter
