"""Measure request latency through the real ASGI surface.

    python scripts/loadtest.py --sessions 20 --messages 5

The model is the scripted stand-in, so these numbers are the platform's own cost
(auth, rate limiting, Postgres, the LangGraph checkpoint, SSE framing) with the
provider removed. They are not model latency and are never presented as quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from langgraph.checkpoint.memory import InMemorySaver

from agent.api.main import create_app
from agent.config import Settings
from agent.db.seed import CUSTOMER_A, CUSTOMER_B
from agent.db.session import create_engine, create_session_factory
from evals.runner import _prepare_schema, _reseed, ensure_database
from evals.scripted import ScriptedModel, say

DEFAULT_EVAL_URL = "postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_eval"
ANSWER = "Hello — I can help with orders, tickets and delivery questions."


@dataclass
class Sample:
    session: int
    index: int
    status: int
    latency_ms: float
    error: str | None = None


@dataclass
class Run:
    samples: list[Sample] = field(default_factory=list)
    wall_seconds: float = 0.0
    baseline: list[Sample] = field(default_factory=list)
    baseline_seconds: float = 0.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--messages", type=int, default=5, help="messages per session")
    parser.add_argument("--out", default="evals/report", help="directory for load.md")
    parser.add_argument("--database-url", default=None)
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings()  # type: ignore[call-arg]
    return base.model_copy(
        update={
            "database_url": args.database_url or DEFAULT_EVAL_URL,
            "checkpoint_dsn": None,
            "environment": "load",
            # The limiter is exercised by its own tests; a budget here would measure
            # 429 latency instead of turn latency.
            "rate_limit_per_min": 1_000_000,
            "rate_limit_burst": 1_000_000,
        }
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):  # pragma: no cover - no git
        return "unknown"


def _command(argv: list[str] | None) -> str:
    parts = ["python", "scripts/loadtest.py", *(argv or sys.argv[1:])]
    return " ".join(parts)


async def _session_run(
    client: httpx.AsyncClient,
    tokens: dict[str, str],
    customer: str,
    index: int,
    args: argparse.Namespace,
) -> list[Sample]:
    headers = {"Authorization": f"Bearer {tokens[customer]}"}
    created = await client.post("/v1/sessions", headers=headers)
    created.raise_for_status()
    session_id = created.json()["session_id"]

    samples: list[Sample] = []
    for message_index in range(args.messages):
        started = time.perf_counter()
        status = 0
        error: str | None = None
        body = ""
        try:
            async with client.stream(
                "POST",
                f"/v1/sessions/{session_id}/messages",
                json={"message": "Hello, what can you do?"},
                headers=headers,
            ) as response:
                status = response.status_code
                async for line in response.aiter_lines():
                    body += line + "\n"
        except httpx.HTTPError as exc:  # pragma: no cover - transport failure
            error = type(exc).__name__
        latency_ms = (time.perf_counter() - started) * 1000
        if status != 200:
            error = error or f"http_{status}"
        elif "event: error" in body:
            error = "stream_error"
        elif "event: done" not in body:
            error = "no_done_event"
        samples.append(
            Sample(
                session=index,
                index=message_index,
                status=status,
                latency_ms=latency_ms,
                error=error,
            )
        )
    return samples


async def _run(args: argparse.Namespace, argv: list[str] | None) -> Run:
    settings = _settings(args)
    ensure_database(settings.database_url)
    engine = create_engine(settings.database_url)
    await _prepare_schema(engine)
    await _reseed(create_session_factory(engine))

    tokens = {
        str(CUSTOMER_A): settings.token_for(CUSTOMER_A) or "",
        str(CUSTOMER_B): settings.token_for(CUSTOMER_B) or "",
    }
    # Half the sessions act as each customer, so identity is exercised under
    # concurrency rather than reused from one token.
    customers = [
        str(CUSTOMER_A) if index % 2 == 0 else str(CUSTOMER_B) for index in range(args.sessions)
    ]

    app = create_app(
        settings,
        model=ScriptedModel(responses=[say(ANSWER)]),
        checkpointer=InMemorySaver(),
    )
    transport = httpx.ASGITransport(app=app)
    run = Run()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://loadtest", timeout=120) as client,
    ):
        # One client alone first: the same code path with no queueing, so the
        # concurrent row can be read as the cost of concurrency rather than as
        # the cost of a turn.
        baseline_started = time.perf_counter()
        run.baseline = await _session_run(client, tokens, str(CUSTOMER_A), -1, args)
        run.baseline_seconds = time.perf_counter() - baseline_started

        started = time.perf_counter()
        results = await asyncio.gather(
            *(
                _session_run(client, tokens, customer, index, args)
                for index, customer in enumerate(customers)
            )
        )
        run.wall_seconds = time.perf_counter() - started
    await engine.dispose()

    for samples in results:
        run.samples.extend(samples)
    return run


def render_markdown(run: Run, args: argparse.Namespace, argv: list[str] | None) -> str:
    latencies = [sample.latency_ms for sample in run.samples]
    failures = [sample for sample in run.samples if sample.error]
    total = len(run.samples)
    error_rate = len(failures) / total if total else 0.0
    throughput = total / run.wall_seconds if run.wall_seconds else 0.0
    baseline_latencies = [sample.latency_ms for sample in run.baseline]
    baseline_throughput = len(run.baseline) / run.baseline_seconds if run.baseline_seconds else 0.0
    lines = [
        "# Load measurement",
        "",
        "Generated by `scripts/loadtest.py` — do not edit by hand.",
        "",
        f"- Command: `{_command(argv)}`",
        f"- Commit: `{_commit()}`",
        f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "- Model: scripted stand-in (`evals/scripted.py`); no provider call is made",
        "",
        "## What this measures",
        "",
        "End-to-end time from `POST /v1/sessions/{id}/messages` to the terminal SSE event,",
        "through the ASGI app: bearer auth, the Postgres-backed rate limiter, session",
        "lookup, the LangGraph turn with an in-memory checkpointer, grounding, and the",
        "SSE framing. It does **not** measure model latency, because no model is called.",
        "Treat these as the platform's own overhead.",
        "",
        "## Results",
        "",
        "| load | requests | wall clock | throughput | p50 | p95 | p99 | errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| 1 client, sequential | {len(run.baseline)} | "
            f"{run.baseline_seconds:.2f} s | {baseline_throughput:.1f} req/s | "
            f"{_percentile(baseline_latencies, 0.50):.0f} ms | "
            f"{_percentile(baseline_latencies, 0.95):.0f} ms | "
            f"{_percentile(baseline_latencies, 0.99):.0f} ms | "
            f"{sum(1 for sample in run.baseline if sample.error)} |"
        )
        if run.baseline
        else "| 1 client, sequential | n/a | n/a | n/a | n/a | n/a | n/a | n/a |",
        (
            f"| {args.sessions} concurrent sessions | {total} | "
            f"{run.wall_seconds:.2f} s | {throughput:.1f} req/s | "
            f"{_percentile(latencies, 0.50):.0f} ms | "
            f"{_percentile(latencies, 0.95):.0f} ms | "
            f"{_percentile(latencies, 0.99):.0f} ms | "
            f"{error_rate:.1%} ({len(failures)}/{total}) |"
        ),
        "",
    ]
    if failures:
        lines += [
            "## Failures",
            "",
            *[
                f"- session {sample.session} message {sample.index}: {sample.error}"
                for sample in failures[:20]
            ],
            "",
        ]
    lines += [
        "## What limits it",
        "",
        "- Every in-flight turn holds one pooled Postgres connection from the first tool",
        "  call to the commit, so concurrent turns are bounded by",
        "  `DB_POOL_SIZE + DB_MAX_OVERFLOW` (10 + 10 by default).",
        "- The tool-bound graph is rebuilt per turn (tools close over the request's",
        "  session), which is a fixed ~20 ms of the turn and does not amortize.",
        "- One process, one database, in-memory checkpointer: no network hop and no",
        "  Postgres checkpoint write. Real deployments are slower, not faster.",
        "- The scripted model returns immediately, so real deployments add provider time",
        "  to every turn (typically hundreds of milliseconds to seconds).",
        "- Run it on the machine you are quoting from; absolute numbers are not portable.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run = asyncio.run(_run(args, argv))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(run, args, argv)
    (out_dir / "load.md").write_text(markdown, encoding="utf-8")
    (out_dir / "load.json").write_text(
        json.dumps(
            {
                "sessions": args.sessions,
                "messages": args.messages,
                "wall_seconds": round(run.wall_seconds, 3),
                "samples": [
                    {
                        "session": sample.session,
                        "index": sample.index,
                        "status": sample.status,
                        "latency_ms": round(sample.latency_ms, 1),
                        "error": sample.error,
                    }
                    for sample in run.samples
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = [sample for sample in run.samples if sample.error]
    latencies = [sample.latency_ms for sample in run.samples]
    print(
        f"{len(run.samples)} requests, {len(failures)} failed, "
        f"p50 {_percentile(latencies, 0.5):.0f} ms, p95 {_percentile(latencies, 0.95):.0f} ms"
    )
    print(f"report written to {out_dir / 'load.md'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
