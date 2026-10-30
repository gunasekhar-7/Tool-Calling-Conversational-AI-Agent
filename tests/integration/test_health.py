from __future__ import annotations

import pytest

from tests.conftest import AUTH


@pytest.mark.integration
async def test_healthz_returns_ok(app_client) -> None:
    async with app_client() as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.integration
async def test_readyz_checks_database(app_client) -> None:
    async with app_client() as client:
        response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.integration
async def test_unknown_session_returns_404(app_client, seeded: None) -> None:
    async with app_client() as client:
        response = await client.get(
            "/v1/sessions/00000000-0000-4000-8000-000000000000/transcript",
            headers=AUTH,
        )
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "not_found"
    assert body["request_id"]


@pytest.mark.integration
async def test_missing_bearer_returns_403(app_client, seeded: None) -> None:
    async with app_client() as client:
        response = await client.get("/v1/customers")
    assert response.status_code == 403
    assert response.json()["code"] == "policy_denied"


@pytest.mark.integration
async def test_request_id_header_is_echoed(app_client, seeded: None) -> None:
    async with app_client() as client:
        response = await client.get("/healthz", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"
