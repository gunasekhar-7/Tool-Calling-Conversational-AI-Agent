from __future__ import annotations

import sys
import types

from agent.config import Settings
from agent.tracing import configure_tracing, flush_tracing


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "database_url": "postgresql://u:p@localhost:5432/db",
        "customer_api_keys": {"key-a": "11111111-1111-4111-8111-111111111111"},
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_tracing_is_a_no_op_without_a_key() -> None:
    settings = _settings()
    assert settings.has_respan_key is False
    assert configure_tracing(settings) is False
    # Flushing without a configured client must not raise either.
    flush_tracing()
    assert "respan" not in sys.modules


def test_tracing_initializes_once_with_the_configured_key(monkeypatch) -> None:
    import agent.tracing as tracing

    calls: list[dict[str, object]] = []
    flushed: list[bool] = []

    class FakeRespan:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def flush(self) -> None:
            flushed.append(True)

    fake = types.ModuleType("respan")
    fake.Respan = FakeRespan
    monkeypatch.setitem(sys.modules, "respan", fake)
    monkeypatch.setattr(tracing, "_tracing", None)

    settings = _settings(respan_api_key="rspn-test")
    assert settings.has_respan_key is True
    assert configure_tracing(settings) is True
    # A second model construction must not double-instrument the provider SDK.
    assert configure_tracing(settings) is True

    assert len(calls) == 1
    assert calls[0]["api_key"] == "rspn-test"
    assert calls[0]["app_name"] == "order-support-agent"

    flush_tracing()
    assert flushed == [True]

    monkeypatch.setattr(tracing, "_tracing", None)
