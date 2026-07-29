"""Customer domain models.

These are internal representations returned by repositories/services. Tool
*output* schemas (in `financial_agent.agent.tools`) are deliberately
separate, narrower models — this keeps the domain free to grow fields that
should never be exposed to the LLM.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RiskProfile(StrEnum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class Product(BaseModel):
    """A financial product the bank offers or the customer already holds."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Stable product identifier, e.g. 'cdb_liquidez_diaria'.")
    name: str = Field(description="Human-readable product name.")
    category: str = Field(description="Product category, e.g. 'renda_fixa', 'seguro'.")
    owned_by_customer: bool = Field(
        default=False, description="Whether the current customer already holds this product."
    )


class CustomerProfile(BaseModel):
    """Aggregated customer view used to ground the agent's recommendations."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    full_name: str
    segment: str = Field(description="CRM segment, e.g. 'private', 'varejo', 'high_income'.")
    risk_profile: RiskProfile
    account_balance: float = Field(ge=0, description="Current checking-account balance in BRL.")
    products: list[Product] = Field(default_factory=list)
    created_at: datetime
