"""Inbound Telegram webhook payload — only the fields this project needs.

`extra="ignore"` on every model: Telegram's `Update` object has dozens of
optional fields (edited_message, callback_query, poll, ...). We deliberately
only model the text-message path and let FastAPI/Pydantic ignore the rest,
so Telegram can evolve its schema without breaking this webhook.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TelegramUser(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    is_bot: bool = False
    first_name: str = ""


class TelegramChat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int


class TelegramMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message_id: int
    from_user: TelegramUser = Field(alias="from")
    chat: TelegramChat
    text: str | None = None


class TelegramUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    update_id: int
    message: TelegramMessage | None = None
