"""Telegram webhook — the application's single entrypoint.

Request flow (mirrors the product spec step by step):

  1. Telegram calls this webhook with an `Update` payload.
  2. `_require_telegram_secret` validates the `X-Telegram-Bot-Api-Secret-Token`
     header (see `security/telegram_auth.py`) before anything else runs.
  3. The body is parsed into a typed `TelegramUpdate` (Pydantic) — non-text
     updates are acknowledged and dropped (Telegram expects a fast 2xx
     regardless of whether we "handled" the update).
  4. `user_id` is taken *only* from `message.from.id` (Telegram's own
     authenticated field) — this is the one and only source of identity for
     the rest of the request; it is never read from the message text.
  5. Per-user rate limiting is enforced before any expensive work happens.
  6. Conversation history is loaded (the agent's memory of this user).
  7. The message is classified as SIMPLE or COMPLEX (`agent/query_complexity.py`)
     to pick the Model Router tier for this turn — cheap/utility model for
     straightforward questions, reasoning model for elaborate or
     loan-related ones.
  8. The main tool-calling agent (`agent/agent_executor.py`) starts as a
     background task. If it hasn't finished within
     `Settings.filler_delay_seconds`, a filler reply is fired (see
     `agent/filler_agent.py`) so the customer sees *something* while the
     agent keeps working — a fast turn never gets a filler at all, so a
     quick question gets exactly one reply, not two back-to-back messages.
  9. The main agent internally calls `get_customer_profile`,
     `get_next_best_action`, `search_knowledge_base`, and/or `request_loan`
     as needed.
  10. Both turns are persisted and the final answer is sent back via the
      Telegram Bot API.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Header
from starlette import status

from financial_agent.agent.agent_executor import build_agent_executor, run_agent_turn
from financial_agent.agent.filler_agent import FillerAgent
from financial_agent.agent.model_router import ModelRouter
from financial_agent.agent.query_complexity import classify_query_complexity
from financial_agent.api.deps import (
    get_conversation_service,
    get_customer_service,
    get_filler_agent,
    get_knowledge_base_service,
    get_loan_service,
    get_model_router,
    get_nba_service,
    get_products_service,
    get_rate_limiter,
    get_settings_dep,
    get_telegram_gateway,
)
from financial_agent.api.schemas.telegram_update import TelegramUpdate
from financial_agent.config import Settings
from financial_agent.domain.errors import UnauthorizedAppError
from financial_agent.domain.models.conversation import MessageRole
from financial_agent.gateways.telegram_gateway import TelegramGateway
from financial_agent.observability.logging import get_logger
from financial_agent.security.rate_limit import RateLimiter
from financial_agent.security.telegram_auth import (
    TELEGRAM_SECRET_HEADER,
    is_valid_telegram_secret,
)
from financial_agent.services.conversation_service import ConversationService
from financial_agent.services.customer_service import CustomerService
from financial_agent.services.knowledge_base_service import KnowledgeBaseService
from financial_agent.services.loan_service import LoanService
from financial_agent.services.nba_service import NBAService
from financial_agent.services.products_service import ProductsService

logger = get_logger(__name__)

router = APIRouter(prefix="/webhook", tags=["telegram"])


def _require_telegram_secret(
    settings: Settings = Depends(get_settings_dep),
    x_telegram_bot_api_secret_token: str | None = Header(
        default=None, alias=TELEGRAM_SECRET_HEADER
    ),
) -> None:
    if not is_valid_telegram_secret(
        received=x_telegram_bot_api_secret_token,
        expected=settings.telegram_webhook_secret,
    ):
        raise UnauthorizedAppError("Invalid Telegram webhook secret token.")


async def _send_filler_reply(
    *, filler_agent: FillerAgent, telegram_gateway: TelegramGateway, chat_id: int, text: str
) -> None:
    """Best-effort: a failure here must never affect the main reply."""
    try:
        await telegram_gateway.send_typing_action(chat_id=chat_id)
        filler = await filler_agent.generate(text)
        await telegram_gateway.send_message(chat_id=chat_id, text=filler.message)
    except Exception as exc:
        logger.warning("filler_reply_failed", chat_id=chat_id, error=str(exc))


@router.post(
    "/telegram",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(_require_telegram_secret)],
)
async def telegram_webhook(
    update: TelegramUpdate,
    settings: Settings = Depends(get_settings_dep),
    telegram_gateway: TelegramGateway = Depends(get_telegram_gateway),
    customer_service: CustomerService = Depends(get_customer_service),
    nba_service: NBAService = Depends(get_nba_service),
    products_service: ProductsService = Depends(get_products_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    loan_service: LoanService = Depends(get_loan_service),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    filler_agent: FillerAgent = Depends(get_filler_agent),
    model_router: ModelRouter = Depends(get_model_router),
) -> dict[str, bool]:
    if update.message is None or not update.message.text:
        # Non-text update (sticker, edited message, etc.) — nothing to do.
        return {"ok": True}

    user_id = update.message.from_user.id
    chat_id = update.message.chat.id
    text = update.message.text

    await rate_limiter.check(str(user_id))

    history = await conversation_service.get_history(user_id, current_message=text)

    complexity = classify_query_complexity(text)
    llm = model_router.for_complexity(complexity)

    executor = build_agent_executor(
        user_id=user_id,
        llm=llm,
        customer_service=customer_service,
        nba_service=nba_service,
        products_service=products_service,
        knowledge_base_service=knowledge_base_service,
        loan_service=loan_service,
    )
    agent_task = asyncio.create_task(
        run_agent_turn(executor=executor, user_message=text, history=history)
    )

    # Only send the filler if the agent is genuinely still working after the
    # delay — a fast turn should get exactly one reply, not a "hold on"
    # message immediately followed by the real answer.
    done, _pending = await asyncio.wait({agent_task}, timeout=settings.filler_delay_seconds)
    filler_task: asyncio.Task[None] | None = None
    if agent_task not in done:
        filler_task = asyncio.create_task(
            _send_filler_reply(
                filler_agent=filler_agent,
                telegram_gateway=telegram_gateway,
                chat_id=chat_id,
                text=text,
            )
        )

    reply_text = await agent_task

    await conversation_service.record_turn(user_id, role=MessageRole.USER, content=text)
    await conversation_service.record_turn(user_id, role=MessageRole.ASSISTANT, content=reply_text)

    await telegram_gateway.send_message(chat_id=chat_id, text=reply_text)
    if filler_task is not None:
        await filler_task  # ensure it has finished (and any error logged) before returning

    logger.info(
        "telegram_turn_completed",
        user_id=user_id,
        complexity=complexity.value,
        filler_sent=filler_task is not None,
    )
    return {"ok": True}
