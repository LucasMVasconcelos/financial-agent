"""FastAPI application entrypoint: `uvicorn financial_agent.main:app`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from financial_agent.api.app_state import build_app_state, shutdown_app_state
from financial_agent.api.middleware.correlation_id import CorrelationIdMiddleware
from financial_agent.api.routers import admin_loans, health, telegram_webhook
from financial_agent.config import get_settings
from financial_agent.domain.errors import AppError
from financial_agent.observability.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, json_output=settings.app_env != "local")

    app.state.container = await build_app_state(settings)
    logger.info("application_started", app_env=settings.app_env)
    try:
        yield
    finally:
        await shutdown_app_state(app.state.container)
        logger.info("application_stopped")


app = FastAPI(
    title="Financial AI Agent",
    description="Telegram-facing Next Best Action financial assistant.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(CorrelationIdMiddleware)


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    """Uniform structured-error JSON body for every `AppError` subclass."""
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": {"code": exc.code.value, "message": exc.message}},
    )


app.include_router(health.router)
app.include_router(telegram_webhook.router)
app.include_router(admin_loans.router)
