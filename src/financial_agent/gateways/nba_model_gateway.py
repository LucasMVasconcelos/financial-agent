"""Next Best Action model gateway.

This is the seam the whole project is designed around: `NBAModelGateway` is
a narrow Protocol (one method, `predict`). Today only `MockNBAModelGateway`
exists, returning a plausible, deterministic-per-customer recommendation.
Swapping in a real model later — a SageMaker endpoint, a Lambda-hosted
model, a plain REST microservice — means writing one new class that
satisfies the same Protocol and flipping `NBA_MODEL_PROVIDER` in config; no
other layer (service, tool, agent, prompt) changes.

`SageMakerNBAModelGateway` is included as a ready-to-use reference
implementation for that swap (see the "AWS" section of the README), guarded
so the optional `boto3` dependency is only required when actually selected.
"""

from __future__ import annotations

import json
import random
from typing import Protocol

from financial_agent.domain.errors import ToolError as DomainToolError
from financial_agent.domain.errors import ToolErrorCode
from financial_agent.domain.models.customer import CustomerProfile
from financial_agent.domain.models.nba import NextBestActionCandidate, NextBestActionType
from financial_agent.observability.logging import get_logger

logger = get_logger(__name__)


class NBAModelGateway(Protocol):
    async def predict(self, customer: CustomerProfile) -> NextBestActionCandidate:
        """Return a recommendation for `customer`.

        Raises:
            ToolError: with code UPSTREAM_ERROR if the model backend fails,
                or RATE_LIMITED if it throttles the caller.
        """
        ...


class MockNBAModelGateway:
    """Deterministic-per-customer fake standing in for a trained NBA model.

    Deterministic (seeded by `user_id`) so demos and tests are reproducible,
    while still varying the recommendation across different customers based
    on simple, explainable heuristics over the profile — close enough to a
    real model's shape (action + confidence + reason) to validate the whole
    pipeline end to end.
    """

    _HIGH_BALANCE_THRESHOLD_BRL = 50_000
    _LOW_BALANCE_THRESHOLD_BRL = 2_000

    async def predict(self, customer: CustomerProfile) -> NextBestActionCandidate:
        owned_codes = {p.code for p in customer.products}

        if (
            customer.account_balance > self._HIGH_BALANCE_THRESHOLD_BRL
            and "cdb_liquidez_diaria" not in owned_codes
        ):
            action = NextBestActionType.INVEST_CDB
            reason = (
                f"Cliente possui saldo elevado (R$ {customer.account_balance:,.2f}) "
                "parado em conta corrente."
            )
            confidence = 0.92
        elif customer.account_balance < self._LOW_BALANCE_THRESHOLD_BRL:
            action = NextBestActionType.BUILD_EMERGENCY_FUND
            reason = "Saldo em conta corrente está abaixo do recomendado para imprevistos."
            confidence = 0.81
        elif "cartao_black" in owned_codes:
            action = NextBestActionType.INCREASE_CREDIT_LIMIT
            reason = "Cliente possui bom histórico de uso do cartão premium."
            confidence = 0.74
        else:
            action = NextBestActionType.INVEST_FIXED_INCOME
            reason = "Perfil de risco compatível com produtos de renda fixa."
            confidence = 0.68

        jitter = random.Random(customer.user_id).uniform(-0.03, 0.03)
        confidence = round(min(max(confidence + jitter, 0.0), 1.0), 2)

        logger.info(
            "nba_prediction_generated",
            user_id=customer.user_id,
            action=action.value,
            confidence=confidence,
        )
        return NextBestActionCandidate(action=action, confidence=confidence, reason=reason)


class SageMakerNBAModelGateway:
    """Reference real-model implementation backed by a SageMaker endpoint.

    Not wired up by default (requires the optional `aws` dependency group:
    `poetry install --with aws`). Selected via `NBA_MODEL_PROVIDER=sagemaker`.
    """

    def __init__(self, *, endpoint_name: str, region_name: str) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "boto3 is required for SageMakerNBAModelGateway; "
                "install with `poetry install --with aws`."
            ) from exc

        self._endpoint_name = endpoint_name
        self._client = boto3.client("sagemaker-runtime", region_name=region_name)

    async def predict(self, customer: CustomerProfile) -> NextBestActionCandidate:
        payload = {
            "user_id": customer.user_id,
            "segment": customer.segment,
            "risk_profile": customer.risk_profile.value,
            "account_balance": customer.account_balance,
            "owned_products": [p.code for p in customer.products],
        }
        try:
            response = self._client.invoke_endpoint(
                EndpointName=self._endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            body = json.loads(response["Body"].read())
            return NextBestActionCandidate.model_validate(body)
        except Exception as exc:
            logger.error("sagemaker_invoke_failed", user_id=customer.user_id, error=str(exc))
            raise DomainToolError(
                ToolErrorCode.UPSTREAM_ERROR, "NBA model backend is unavailable."
            ) from exc
