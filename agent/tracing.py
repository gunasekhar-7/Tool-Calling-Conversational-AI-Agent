"""Respan tracing at the model seam.

Tracing is opt-in and decided by configuration: with ``RESPAN_API_KEY`` set,
Respan is initialized once before the first model client is built, so the OpenAI
SDK calls that client makes are exported as traces. Without the key this module
imports nothing, exports nothing, and the app runs exactly as it did before.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.config import Settings

logger = logging.getLogger(__name__)

# Module-level because Respan patches the provider SDK globally: a second client
# would double-instrument every call.
_tracing: Any | None = None


def configure_tracing(settings: Settings) -> bool:
    """Initialize Respan once. Returns whether tracing is active."""
    global _tracing
    if _tracing is not None:
        return True
    if not settings.has_respan_key:
        return False

    assert settings.respan_api_key is not None
    from respan import Respan

    _tracing = Respan(
        api_key=settings.respan_api_key.get_secret_value(),
        app_name="order-support-agent",
        environment=settings.environment,
        metadata={"service": "order-support-agent"},
    )
    logger.info(
        "respan_tracing_enabled",
        extra={"extra_fields": {"environment": settings.environment}},
    )
    return True


def flush_tracing() -> None:
    """Export anything still buffered; a short-lived process needs this before exit."""
    if _tracing is not None:
        _tracing.flush()
