from __future__ import annotations

import secrets

from fastapi import Request

from agent.config import Settings
from agent.errors import PolicyDenied
from agent.runtime import AgentRuntime


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise PolicyDenied("service is not ready")
    return runtime


def require_api_key(request: Request) -> None:
    settings: Settings = request.app.state.settings
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise PolicyDenied("missing bearer token")
    if not secrets.compare_digest(token, settings.demo_api_key):
        raise PolicyDenied("invalid bearer token")
