from __future__ import annotations

import secrets
from dataclasses import dataclass
from uuid import UUID

from fastapi import Request

from agent.config import Settings
from agent.errors import PolicyDenied
from agent.runtime import AgentRuntime


@dataclass(frozen=True)
class Principal:
    """The authenticated caller.

    ``customer_id`` comes from configuration, never from a request body or the
    model. That is what makes every tool call authorized by construction.
    """

    customer_id: UUID
    token_label: str


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise PolicyDenied("service is not ready")
    return runtime


def _matches(token: str, candidate: str) -> bool:
    try:
        return secrets.compare_digest(token, candidate)
    except TypeError:  # non-ASCII token: compare_digest cannot take it
        return False


def resolve_principal(settings: Settings, authorization: str) -> Principal:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise PolicyDenied("missing bearer token")

    # Compare against every configured token so a wrong token costs the same as a
    # missing one, regardless of which customer it is closest to.
    matched: UUID | None = None
    for candidate_token, customer_id in settings.customer_api_keys.items():
        if _matches(token.strip(), candidate_token):
            matched = customer_id
    if matched is None:
        raise PolicyDenied("invalid bearer token")
    return Principal(customer_id=matched, token_label=f"customer:{matched}")


def require_principal(request: Request) -> Principal:
    settings: Settings = request.app.state.settings
    return resolve_principal(settings, request.headers.get("authorization", ""))
