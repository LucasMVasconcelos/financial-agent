"""API tests for the admin loan approval endpoints.

Covers auth (X-Service-Api-Key), listing pending applications, deciding
(approve/reject), and the two failure paths a human reviewer can hit:
unknown application id and an application that was already decided.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from financial_agent.api.app_state import AppState
from financial_agent.security.auth import SERVICE_API_KEY_HEADER

VALID_KEY = "test-service-key"
KNOWN_USER_ID = 123


@pytest.fixture(autouse=True)
def _stub_telegram(app_state: AppState) -> None:
    app_state.telegram_gateway.send_message = AsyncMock()  # type: ignore[method-assign]


class TestAdminLoansAuth:
    async def test_missing_api_key_returns_401(self, client: AsyncClient) -> None:
        response = await client.get("/admin/loans")
        assert response.status_code == 401

    async def test_invalid_api_key_returns_401(self, client: AsyncClient) -> None:
        response = await client.get(
            "/admin/loans", headers={SERVICE_API_KEY_HEADER: "wrong-key"}
        )
        assert response.status_code == 401


class TestAdminLoansList:
    async def test_lists_pending_applications(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        application = await app_state.loan_service.request_loan(
            user_id=KNOWN_USER_ID, amount=80_000.0
        )

        response = await client.get(
            "/admin/loans", headers={SERVICE_API_KEY_HEADER: VALID_KEY}
        )

        assert response.status_code == 200
        body = response.json()
        assert [a["application_id"] for a in body] == [application.application_id]
        assert body[0]["requires_human_approval"] is True

    async def test_auto_approved_applications_are_not_in_pending_list(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        await app_state.loan_service.request_loan(user_id=KNOWN_USER_ID, amount=1_000.0)

        response = await client.get(
            "/admin/loans", headers={SERVICE_API_KEY_HEADER: VALID_KEY}
        )

        assert response.json() == []


class TestAdminLoansDecide:
    async def test_approve_disburses_and_notifies_customer(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        application = await app_state.loan_service.request_loan(
            user_id=KNOWN_USER_ID, amount=80_000.0
        )

        response = await client.post(
            f"/admin/loans/{application.application_id}/decide",
            json={"approved": True, "decided_by": "ana.analista"},
            headers={SERVICE_API_KEY_HEADER: VALID_KEY},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "disbursed"
        assert body["decided_by"] == "ana.analista"
        app_state.telegram_gateway.send_message.assert_awaited_once()  # type: ignore[attr-defined]

    async def test_reject_updates_status(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        application = await app_state.loan_service.request_loan(
            user_id=KNOWN_USER_ID, amount=80_000.0
        )

        response = await client.post(
            f"/admin/loans/{application.application_id}/decide",
            json={"approved": False, "decided_by": "ana.analista"},
            headers={SERVICE_API_KEY_HEADER: VALID_KEY},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    async def test_unknown_application_returns_404(self, client: AsyncClient) -> None:
        response = await client.post(
            "/admin/loans/does-not-exist/decide",
            json={"approved": True, "decided_by": "ana"},
            headers={SERVICE_API_KEY_HEADER: VALID_KEY},
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    async def test_already_decided_application_returns_422(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        application = await app_state.loan_service.request_loan(
            user_id=KNOWN_USER_ID, amount=1_000.0
        )  # auto-approved, not pending

        response = await client.post(
            f"/admin/loans/{application.application_id}/decide",
            json={"approved": True, "decided_by": "ana"},
            headers={SERVICE_API_KEY_HEADER: VALID_KEY},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_decide_requires_api_key(
        self, client: AsyncClient, app_state: AppState
    ) -> None:
        application = await app_state.loan_service.request_loan(
            user_id=KNOWN_USER_ID, amount=80_000.0
        )

        response = await client.post(
            f"/admin/loans/{application.application_id}/decide",
            json={"approved": True, "decided_by": "ana"},
        )

        assert response.status_code == 401
