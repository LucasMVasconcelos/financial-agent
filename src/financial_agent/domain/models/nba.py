"""Next Best Action (NBA) domain models.

`NextBestActionCandidate` is what the ML gateway (mock today, SageMaker
tomorrow — see `financial_agent.gateways.nba_model_gateway`) returns. It is
intentionally identical in shape to the Tool output schema
(`GetNextBestActionOutput`) today, but the two are kept as separate types so
the Tool contract can evolve independently of the model's raw output.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class NextBestActionType(StrEnum):
    INVEST_FIXED_INCOME = "investir_em_renda_fixa"
    INCREASE_CREDIT_LIMIT = "aumentar_limite_do_cartao"
    HIRE_INSURANCE = "contratar_seguro"
    ANTICIPATE_INSTALLMENTS = "antecipar_parcelas"
    PORT_DEBT = "fazer_portabilidade"
    INVEST_CDB = "aplicar_em_cdb"
    BUILD_EMERGENCY_FUND = "criar_reserva_de_emergencia"


class NextBestActionCandidate(BaseModel):
    """Raw recommendation produced by the NBA model gateway."""

    model_config = ConfigDict(frozen=True)

    action: NextBestActionType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Short, factual justification grounded in customer data.")
