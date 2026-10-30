from __future__ import annotations

from uuid import UUID

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.api.main import create_app
from agent.config import Settings
from agent.db.seed import CUSTOMER_A, CUSTOMER_B
from tests.conftest import AUTH, AUTH_B, CUSTOMER_TOKENS, create_session, running_client

pytestmark = pytest.mark.integration

ANON = {"Authorization": "Bearer not-a-configured-token"}


async def test_unknown_token_is_denied(app_client) -> None:
    async with app_client(None) as client:
        response = await client.post("/v1/sessions", headers=ANON)
    assert response.status_code == 403
    assert response.json()["code"] == "policy_denied"


async def test_session_body_cannot_choose_the_customer(app_client) -> None:
    async with app_client(None) as client:
        response = await client.post(
            "/v1/sessions",
            json={"customer_id": str(CUSTOMER_B)},
            headers=AUTH,
        )
    assert response.status_code == 201
    assert response.json()["customer_id"] == str(CUSTOMER_A)


async def test_customer_listing_is_scoped_to_the_caller(app_client) -> None:
    async with app_client(None) as client:
        mine = await client.get("/v1/customers", headers=AUTH)
        theirs = await client.get("/v1/customers", headers=AUTH_B)
    assert [row["id"] for row in mine.json()] == [str(CUSTOMER_A)]
    assert [row["id"] for row in theirs.json()] == [str(CUSTOMER_B)]


@pytest.mark.parametrize(
    ("method", "suffix", "payload"),
    [
        ("GET", "", None),
        ("GET", "/transcript", None),
        ("POST", "/messages", {"message": "Hello?"}),
        (
            "POST",
            "/approvals",
            {
                "pending_action_id": "00000000-0000-4000-8000-000000000000",
                "decision": "approve",
                "idempotency_key": "cross-tenant-key",
                "preview_hash": "0" * 64,
            },
        ),
    ],
)
async def test_another_customers_session_is_invisible(
    app_client, method: str, suffix: str, payload: dict | None
) -> None:
    async with app_client(None) as client:
        session_id = await create_session(client, str(CUSTOMER_A))
        response = await client.request(
            method,
            f"/v1/sessions/{session_id}{suffix}",
            json=payload,
            headers=AUTH_B,
        )
    # 404, never 403: the API must not confirm that the conversation exists.
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_rotated_key_rejects_the_old_token_and_keeps_the_session(
    settings: Settings,
) -> None:
    async with running_client(
        lambda: create_app(settings, model=None, checkpointer=InMemorySaver())
    ) as client:
        session_id = await create_session(client, str(CUSTOMER_A))

    rotated = settings.model_copy(
        update={"customer_api_keys": {"rotated-token": UUID(str(CUSTOMER_A))}}
    )
    async with running_client(
        lambda: create_app(rotated, model=None, checkpointer=InMemorySaver())
    ) as client:
        old = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
        new = await client.get(
            f"/v1/sessions/{session_id}",
            headers={"Authorization": "Bearer rotated-token"},
        )

    assert old.status_code == 403
    assert new.status_code == 200
    assert new.json()["customer_id"] == str(CUSTOMER_A)


async def test_token_map_is_not_a_shared_secret() -> None:
    """Every token resolves to exactly one customer; none is a wildcard."""
    assert set(CUSTOMER_TOKENS) == {str(CUSTOMER_A), str(CUSTOMER_B)}
    assert len(set(CUSTOMER_TOKENS.values())) == len(CUSTOMER_TOKENS)


async def test_no_authorization_header_is_denied(app_client) -> None:
    async with app_client(None) as client:
        response: httpx.Response = await client.post("/v1/sessions")
    assert response.status_code == 403
