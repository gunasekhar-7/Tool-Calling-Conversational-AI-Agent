from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# Buckets chosen around a tool-calling turn: a read is fast, a model turn is not.
TURN_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


class Metrics:
    """One registry per application instance.

    Per-instance rather than module-global so a test can assert exact values and
    a second app in the same process cannot inherit another's counts.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "agent_http_requests_total",
            "HTTP requests by matched route, method and status.",
            ["route", "method", "status"],
            registry=self.registry,
        )
        self.turn_seconds = Histogram(
            "agent_turn_seconds",
            "Wall time for one user message to reach its terminal event.",
            ["outcome"],
            buckets=TURN_BUCKETS,
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "agent_tool_calls_total",
            "Tool invocations by tool and outcome.",
            ["tool", "outcome"],
            registry=self.registry,
        )
        self.approvals = Counter(
            "agent_approvals_total",
            "Human decisions on staged writes.",
            ["decision"],
            registry=self.registry,
        )
        self.guardrail_blocks = Counter(
            "agent_guardrail_blocks_total",
            "Requests stopped by a guardrail before (or instead of) a model call.",
            ["reason"],
            registry=self.registry,
        )

    def observe_request(self, route: str, method: str, status: int) -> None:
        self.http_requests.labels(route=route, method=method, status=str(status)).inc()

    def observe_tool(self, tool: str, outcome: str) -> None:
        self.tool_calls.labels(tool=tool, outcome=outcome).inc()

    def observe_approval(self, decision: str) -> None:
        self.approvals.labels(decision=decision).inc()

    def observe_block(self, reason: str) -> None:
        self.guardrail_blocks.labels(reason=reason).inc()

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def record_tool(ctx: Any, tool: str, outcome: str) -> None:
    """Record against the context's metrics when one is attached."""
    metrics = getattr(ctx, "metrics", None)
    if isinstance(metrics, Metrics):
        metrics.observe_tool(tool, outcome)
