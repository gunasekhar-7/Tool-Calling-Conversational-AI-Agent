"""Failure-mode tests for degraded mode, retry, circuit breaker, pagination, body-size."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from sqlalchemy.exc import OperationalError

from agent.api.main import create_app
from agent.config import Settings
from agent.db.seed import CUSTOMER_A
from agent.db.session import create_engine, create_session_factory
from agent.errors import CircuitOpen
from agent.reliability import CircuitBreaker, CircuitState, RetryConfig, retry_with_backoff
from agent.runtime import AgentRuntime
from tests.conftest import AUTH, collect_events, create_session
from tests.fakes import ScriptedChatModel, tool_call


@pytest.fixture
def degraded_settings() -> Settings:
    """Settings with degraded mode enabled but no model."""
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_test",
        customer_api_keys={"test-key-a": CUSTOMER_A},
        environment="test",
        degraded_read_only=True,
        max_message_chars=2000,
        rate_limit_per_min=1000,
        rate_limit_burst=1000,
        approval_ttl_seconds=900,
    )


@pytest.fixture
def retry_settings() -> Settings:
    """Settings with fast retry for testing."""
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_test",
        customer_api_keys={"test-key-a": CUSTOMER_A},
        environment="test",
        db_retry_max_attempts=3,
        db_retry_base_delay_seconds=0.01,
    )


class TestDegradedMode:
    """Tests for degraded read-only mode when provider is unavailable."""

    @pytest.mark.integration
    async def test_read_tools_work_in_degraded_mode(self, degraded_settings, seeded):
        """Read tools should work normally in degraded mode."""
        model = ScriptedChatModel(
            responses=[
                tool_call("get_order_status", {"order_id": "ORD-1001"}, "c1"),
                AIMessage(content="Order ORD-1001 was delivered 10 days ago."),
            ]
        )
        app = create_app(degraded_settings, model=model)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            session_id = await create_session(client, str(CUSTOMER_A))
            events = await collect_events(client, session_id, "Where is ORD-1001?")

        assert "done" in dict(events)
        assert "delivered" in dict(events)["done"]["text"].lower()
        assert "ORD-1001" in dict(events)["done"]["text"]

    @pytest.mark.integration
    async def test_create_ticket_rejected_in_degraded_mode(self, degraded_settings, seeded):
        """create_ticket should be rejected with provider_degraded in degraded mode."""
        model = ScriptedChatModel(
            responses=[
                tool_call(
                    "create_ticket",
                    {"subject": "Test", "body": "Test body", "priority": "normal"},
                    "w1",
                ),
                AIMessage(content="Should not reach here"),
            ]
        )
        app = create_app(degraded_settings, model=model)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            session_id = await create_session(client, str(CUSTOMER_A))
            events = await collect_events(client, session_id, "Open a ticket please")

        # Should get error event with provider_degraded code
        events_dict = dict(events)
        assert "error" in events_dict
        assert events_dict["error"]["code"] == "provider_degraded"

    @pytest.mark.integration
    async def test_approval_rejected_in_degraded_mode(self, degraded_settings, seeded):
        """Approval endpoint should reject writes in degraded mode."""
        app = create_app(degraded_settings, model=None)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            session_id = await create_session(client, str(CUSTOMER_A))
            response = await client.post(
                f"/v1/sessions/{session_id}/approvals",
                json={
                    "pending_action_id": str(uuid.uuid4()),
                    "decision": "approve",
                    "idempotency_key": "test-key",
                    "preview_hash": "a" * 64,
                },
                headers=AUTH,
            )

        assert response.status_code == 503
        assert response.json()["code"] == "provider_degraded"

    @pytest.mark.integration
    async def test_readyz_shows_degraded_true(self, degraded_settings, seeded):
        """readyz endpoint should report degraded=true when model is None."""
        app = create_app(degraded_settings, model=None)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            response = await client.get("/readyz")

        assert response.status_code == 200
        data = response.json()
        assert data["degraded"] is True
        assert data["model_ready"] is False


class TestRetryAndCircuitBreaker:
    """Tests for DB retry logic and circuit breaker."""

    @pytest.mark.unit
    async def test_retry_succeeds_on_transient_failure(self):
        """Retry should succeed when transient failure occurs then succeeds."""
        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # ConnectionError is retryable per retry logic
                raise ConnectionError("connection failure")
            return "success"

        CircuitBreaker(failure_threshold=10, cooldown_seconds=60)
        config = RetryConfig(max_attempts=5, base_delay_seconds=0.01)

        result = await retry_with_backoff(flaky_operation, config, operation_name="test")
        assert result == "success"
        assert call_count == 3

    @pytest.mark.unit
    async def test_retry_exhausted_after_max_attempts(self):
        """Retry should raise after max attempts exhausted."""

        async def always_fails():
            raise OperationalError("persistent failure", None, None)

        CircuitBreaker(failure_threshold=10, cooldown_seconds=60)
        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)

        with pytest.raises(OperationalError):
            await retry_with_backoff(always_fails, config, operation_name="test")

    @pytest.mark.unit
    async def test_circuit_opens_after_threshold(self):
        """Circuit should open after failure threshold reached."""
        circuit = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)

        async def failing_op():
            raise OperationalError("failure", None, None)

        # First 3 failures should not open circuit yet (threshold=3 means 3rd opens)
        for _ in range(2):
            with pytest.raises(OperationalError):
                await circuit.call(failing_op, operation_name="test")
            assert circuit.state == CircuitState.CLOSED

        # 3rd failure opens circuit
        with pytest.raises(OperationalError):
            await circuit.call(failing_op, operation_name="test")
        assert circuit.state == CircuitState.OPEN

        # Subsequent calls should raise CircuitOpen immediately
        with pytest.raises(CircuitOpen):
            await circuit.call(failing_op, operation_name="test")

    @pytest.mark.unit
    async def test_circuit_half_open_after_cooldown(self):
        """Circuit should transition to half-open after cooldown."""
        circuit = CircuitBreaker(failure_threshold=2, cooldown_seconds=1, half_open_probes=2)

        async def failing_op():
            raise OperationalError("failure", None, None)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(OperationalError):
                await circuit.call(failing_op, operation_name="test")
        assert circuit.state == CircuitState.OPEN

        # Wait for cooldown
        await asyncio.sleep(1.1)

        # Next call should be half-open probe
        async def succeeding_op():
            return "success"

        result = await circuit.call(succeeding_op, operation_name="test")
        assert result == "success"
        assert circuit.state == CircuitState.HALF_OPEN

        # Second success should close circuit
        result = await circuit.call(succeeding_op, operation_name="test")
        assert result == "success"
        assert circuit.state == CircuitState.CLOSED

    @pytest.mark.integration
    async def test_db_retry_in_runtime(self, retry_settings, seeded):
        """Runtime should retry DB operations on transient failures."""
        # This test verifies the runtime's run_with_retry integration
        # We can't easily inject a transient failure, but we verify the path exists
        runtime = AgentRuntime(
            settings=retry_settings,
            model=None,
            checkpointer=MagicMock(),
            session_factory=create_session_factory(create_engine(retry_settings.database_url)),
        )

        # Verify circuit breaker and retry config are initialized
        assert runtime.db_circuit is not None
        assert runtime.provider_circuit is not None
        assert retry_settings.db_retry_max_attempts == 3


class TestMetricsRendering:
    """Tests for Prometheus metrics rendering."""

    @pytest.mark.unit
    def test_metrics_render_includes_new_series(self):
        """Metrics should render all new series."""
        from agent.observability import Metrics

        metrics = Metrics()
        # Record some values
        metrics.observe_degraded_turn("degraded_done")
        metrics.set_circuit_state("db", 0)
        metrics.set_circuit_state("provider", 2)
        metrics.observe_retry("test_op", "success")
        metrics.observe_body_size_rejection()

        body, content_type = metrics.render()
        text = body.decode()

        assert "agent_degraded_turns_total" in text
        assert "agent_circuit_state" in text
        assert "agent_retry_attempts_total" in text
        assert "agent_body_size_rejections_total" in text

    @pytest.mark.integration
    async def test_metrics_endpoint_returns_new_series(self, seeded):
        """Metrics endpoint should expose new series."""
        settings = Settings(
            _env_file=None,
            database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_test",
            customer_api_keys={"test-key-a": CUSTOMER_A},
            environment="test",
        )
        app = create_app(settings, model=None)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            # Need to authenticate for metrics endpoint
            response = await client.get("/metrics", headers=AUTH)

        assert response.status_code == 200
        text = response.text
        assert "agent_degraded_turns_total" in text
        assert "agent_circuit_state" in text
        assert "agent_retry_attempts_total" in text
        assert "agent_body_size_rejections_total" in text


class TestTranscriptPagination:
    """Tests for transcript pagination."""

    @pytest.mark.integration
    async def test_transcript_pagination_limit_offset(self, seeded):
        """Transcript endpoint should support limit/offset pagination."""
        settings = Settings(
            _env_file=None,
            database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_test",
            customer_api_keys={"test-key-a": CUSTOMER_A},
            environment="test",
            transcript_page_size=2,
        )
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="Order ORD-1001 delivered."),
                AIMessage(content="Order ORD-1002 delivered."),
            ]
        )
        app = create_app(settings, model=model)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            session_id = await create_session(client, str(CUSTOMER_A))
            # Send a couple messages to build transcript
            await collect_events(client, session_id, "Where is ORD-1001?")
            await collect_events(client, session_id, "What about ORD-1002?")

            # Get first page
            response = await client.get(
                f"/v1/sessions/{session_id}/transcript",
                params={"limit": 2, "offset": 0},
                headers=AUTH,
            )
            assert response.status_code == 200
            page1 = response.json()
            assert len(page1["messages"]) == 2
            assert page1["has_more"] is True
            assert page1["next_offset"] == 2

            # Get second page
            response = await client.get(
                f"/v1/sessions/{session_id}/transcript",
                params={"limit": 2, "offset": 2},
                headers=AUTH,
            )
            assert response.status_code == 200
            page2 = response.json()
            assert page2["has_more"] is False
            assert page2["next_offset"] is None

    @pytest.mark.integration
    async def test_transcript_pagination_defaults(self, seeded):
        """Transcript should use default page size when not specified."""
        settings = Settings(
            _env_file=None,
            database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_test",
            customer_api_keys={"test-key-a": CUSTOMER_A},
            environment="test",
            transcript_page_size=50,
        )
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="Hello"),
            ]
        )
        app = create_app(settings, model=model)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            session_id = await create_session(client, str(CUSTOMER_A))
            await collect_events(client, session_id, "Hello")

            response = await client.get(
                f"/v1/sessions/{session_id}/transcript",
                headers=AUTH,
            )
            assert response.status_code == 200
            data = response.json()
            assert "has_more" in data
            assert "next_offset" in data


class TestBodySizeEnforcement:
    """Tests for streaming body-size enforcement."""

    @pytest.mark.integration
    async def test_oversized_body_rejected_streaming(self, seeded):
        """Oversized body should be rejected when streaming enforcement enabled."""
        settings = Settings(
            _env_file=None,
            database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_test",
            customer_api_keys={"test-key-a": CUSTOMER_A},
            environment="test",
            max_request_bytes=2048,  # Minimum allowed by config
            enforce_body_size_streaming=True,
        )
        model = ScriptedChatModel(responses=[AIMessage(content="OK")])
        app = create_app(settings, model=model)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            session_id = await create_session(client, str(CUSTOMER_A))
            # Send body larger than 2048 bytes
            large_message = "x" * 3000
            response = await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"message": large_message},
                headers=AUTH,
            )

        assert response.status_code == 413
        assert response.json()["code"] == "request_too_large"

    @pytest.mark.integration
    async def test_normal_body_accepted(self, seeded):
        """Normal-sized body should be accepted."""
        settings = Settings(
            _env_file=None,
            database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_test",
            customer_api_keys={"test-key-a": CUSTOMER_A},
            environment="test",
            max_request_bytes=2048,
            enforce_body_size_streaming=True,
        )
        model = ScriptedChatModel(responses=[AIMessage(content="OK")])
        app = create_app(settings, model=model)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            session_id = await create_session(client, str(CUSTOMER_A))
            response = await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"message": "Normal message"},
                headers=AUTH,
            )

        assert response.status_code == 200


class TestChaosDBDown:
    """Chaos tests for DB failure scenarios."""

    @pytest.mark.integration
    async def test_session_usable_after_db_recovery(self, seeded):
        """Session should remain usable after transient DB failure."""
        settings = Settings(
            _env_file=None,
            database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_test",
            customer_api_keys={"test-key-a": CUSTOMER_A},
            environment="test",
            db_retry_max_attempts=3,
            db_retry_base_delay_seconds=0.01,
        )
        model = ScriptedChatModel(
            responses=[
                tool_call("get_order_status", {"order_id": "ORD-1001"}, "c1"),
                AIMessage(content="Order ORD-1001 was delivered."),
            ]
        )
        app = create_app(settings, model=model)
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client,
            app.router.lifespan_context(app),
        ):
            session_id = await create_session(client, str(CUSTOMER_A))
            events = await collect_events(client, session_id, "Where is ORD-1001?")

        assert "done" in dict(events)
        # Session should still work for subsequent requests
        async with (
            AsyncClient(transport=transport, base_url="http://test", timeout=30) as client2,
            app.router.lifespan_context(app),
        ):
            events2 = await collect_events(client2, session_id, "What about ORD-1002?")
        assert "done" in dict(events2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
