from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from agent.api.main import create_app
from agent.config import Settings
from agent.db.models import Base, PendingAction, Ticket
from agent.db.seed import CUSTOMER_B, reset_and_seed
from agent.db.session import create_session_factory
from evals.scenarios import Scenario, Turn


@dataclass
class TurnOutcome:
    index: int
    status: int
    tools: list[str]
    reply: str
    error_code: str | None
    confirmation: dict[str, Any] | None
    approval_status: int | None
    approval_body: dict[str, Any] | None
    follow_up_status: int | None
    follow_up_body: dict[str, Any] | None
    latency_ms: float
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "tools": self.tools,
            "error_code": self.error_code,
            "latency_ms": round(self.latency_ms, 1),
            "failures": list(self.failures),
        }


@dataclass
class TrialResult:
    scenario_id: str
    category: str
    trial: int
    passed: bool
    turns: list[TurnOutcome]
    failures: list[str]

    @property
    def latencies(self) -> list[float]:
        return [turn.latency_ms for turn in self.turns]


# --------------------------------------------------------------------------- #
# database helpers
# --------------------------------------------------------------------------- #


def _dsn_parts(url: str) -> tuple[str, str]:
    """Split (admin_dsn, database_name) for a postgresql+psycopg URL."""
    plain = url.replace("postgresql+psycopg://", "postgresql://")
    scheme, rest = plain.split("://", 1)
    host_and_db, _, _query = rest.partition("?")
    base, _, database = host_and_db.rpartition("/")
    return f"{scheme}://{base}/postgres", database


def ensure_database(url: str) -> None:
    admin, database = _dsn_parts(url)
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=5) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (database,)
            ).fetchone()
            if exists is None:
                conn.execute(f'CREATE DATABASE "{database}"')
    except psycopg.OperationalError as exc:  # pragma: no cover - environment failure
        raise SystemExit(f"PostgreSQL is not reachable for the evaluation run: {exc}") from exc


async def _prepare_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _reseed(factory: async_sessionmaker) -> None:
    async with factory() as db:
        await reset_and_seed(db)
        await db.commit()


async def _ticket_ids(factory: async_sessionmaker, customer_id: uuid.UUID) -> set[str]:
    async with factory() as db:
        rows = await db.scalars(select(Ticket).where(Ticket.customer_id == customer_id))
        return {row.id for row in rows}


async def _pending_status(factory: async_sessionmaker, session_id: uuid.UUID) -> str | None:
    async with factory() as db:
        row = await db.scalar(
            select(PendingAction)
            .where(PendingAction.session_id == session_id)
            .order_by(PendingAction.created_at.desc())
            .limit(1)
        )
        return row.status if row else None


async def _ticket(factory: async_sessionmaker, ticket_id: str) -> Ticket | None:
    async with factory() as db:
        return await db.scalar(select(Ticket).where(Ticket.id == ticket_id))


# --------------------------------------------------------------------------- #
# transport helpers
# --------------------------------------------------------------------------- #


async def _collect(
    client: httpx.AsyncClient, session_id: uuid.UUID, message: str, headers: dict[str, str]
) -> tuple[int, list[tuple[str, dict[str, Any]]], str | None]:
    """Drive one user message and return (status, events, error_code).

    Rejections happen before the stream opens, so the error code lives in a JSON
    body rather than in an ``error`` event. Both shapes are graded the same way.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    name = ""
    async with client.stream(
        "POST",
        f"/v1/sessions/{session_id}/messages",
        json={"message": message},
        headers=headers,
    ) as response:
        if response.status_code != 200:
            raw = await response.aread()
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            return response.status_code, events, body.get("code")
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[len("data: ") :])))
    return 200, events, None


def _last(events: list[tuple[str, dict[str, Any]]], name: str) -> dict[str, Any] | None:
    found = None
    for event, data in events:
        if event == name:
            found = data
    return found


def _tool_sequence(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [data["name"] for event, data in events if event == "tool_call"]


def _is_ordered_subsequence(expected: tuple[str, ...], actual: list[str]) -> bool:
    iterator = iter(actual)
    return all(any(candidate == want for candidate in iterator) for want in expected)


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #


def _grade_turn(turn: Turn, outcome: TurnOutcome) -> None:
    def fail(message: str) -> None:
        outcome.failures.append(message)

    if outcome.status != turn.expect_status:
        fail(f"expected HTTP {turn.expect_status}, got {outcome.status}")

    if turn.expect_status != 200:
        if outcome.error_code != turn.expect_error_code:
            fail(f"expected error code {turn.expect_error_code}, got {outcome.error_code}")
        return

    if turn.approval is None:
        if outcome.confirmation is not None:
            fail("a write was staged without the scenario expecting confirmation")
        if outcome.reply == "":
            fail("no final answer was produced")
    else:
        if outcome.confirmation is None:
            fail("confirmation_required event was never emitted")
        else:
            if not outcome.confirmation.get("pending_action_id"):
                fail("confirmation payload has no pending_action_id")
            if not outcome.confirmation.get("preview_hash"):
                fail("confirmation payload has no preview_hash")
            if outcome.approval_status != 200:
                fail(f"approval returned HTTP {outcome.approval_status}")
            elif (
                turn.approval == "approve"
                and (outcome.approval_body or {}).get("status") != "executed"
            ):
                fail(f"approval did not execute: {outcome.approval_body}")

    if turn.expect_tools and not _is_ordered_subsequence(turn.expect_tools, outcome.tools):
        fail(f"expected tools {list(turn.expect_tools)} in order, saw {outcome.tools}")

    for forbidden in turn.forbid_tools:
        if forbidden in outcome.tools:
            fail(f"tool {forbidden} must not be called")

    lowered = outcome.reply.lower()
    for wanted in turn.reply_contains:
        if wanted.lower() not in lowered:
            fail(f"reply missing {wanted!r}")
    for unwanted in turn.reply_excludes:
        if unwanted.lower() in lowered:
            fail(f"reply contains forbidden {unwanted!r}")

    if outcome.follow_up_status is not None:
        if turn.follow_up_key == "same":
            if outcome.follow_up_status != 200:
                fail(f"replayed approval should return 200, got {outcome.follow_up_status}")
            elif not (outcome.follow_up_body or {}).get("replayed"):
                fail("replayed approval did not report replayed=true")
        elif outcome.follow_up_status != 409:
            fail(f"second decision should conflict (409), got {outcome.follow_up_status}")


async def run_trial(
    scenario: Scenario,
    *,
    settings: Settings,
    model: Any,
    engine: AsyncEngine,
    trial: int,
) -> TrialResult:
    factory = create_session_factory(engine)
    await _reseed(factory)
    before_other = await _ticket_ids(factory, CUSTOMER_B)
    before_own = await _ticket_ids(factory, scenario.customer)

    app = create_app(settings, model=model, checkpointer=InMemorySaver())
    token = settings.token_for(scenario.customer)
    if token is None:
        raise SystemExit(
            f"CUSTOMER_API_KEYS has no token for scenario customer {scenario.customer}"
        )
    headers = {"Authorization": f"Bearer {token}"}
    outcomes: list[TurnOutcome] = []

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://eval", timeout=60
        ) as client:
            created = await client.post("/v1/sessions", headers=headers)
            if created.status_code != 201:
                return TrialResult(
                    scenario_id=scenario.id,
                    category=scenario.category,
                    trial=trial,
                    passed=False,
                    turns=[],
                    failures=[f"session creation failed: {created.status_code} {created.text}"],
                )
            session_id = uuid.UUID(created.json()["session_id"])

            for index, turn in enumerate(scenario.turns):
                started = time.perf_counter()
                status, events, http_error = await _collect(
                    client, session_id, turn.message, headers
                )
                latency_ms = (time.perf_counter() - started) * 1000
                error = _last(events, "error")
                done = _last(events, "done")
                outcome = TurnOutcome(
                    index=index,
                    status=status,
                    tools=_tool_sequence(events),
                    reply=(done or {}).get("text", ""),
                    error_code=http_error or (error or {}).get("code"),
                    confirmation=_last(events, "confirmation_required"),
                    approval_status=None,
                    approval_body=None,
                    follow_up_status=None,
                    follow_up_body=None,
                    latency_ms=latency_ms,
                )

                if turn.approval is not None and outcome.confirmation is not None:
                    key = f"{scenario.id}-t{trial}-turn{index}-{uuid.uuid4().hex[:8]}"
                    payload = {
                        "pending_action_id": outcome.confirmation["pending_action_id"],
                        "decision": turn.approval,
                        "idempotency_key": key,
                        "preview_hash": outcome.confirmation["preview_hash"],
                    }
                    response = await client.post(
                        f"/v1/sessions/{session_id}/approvals", json=payload, headers=headers
                    )
                    outcome.approval_status = response.status_code
                    outcome.approval_body = response.json()
                    # The turn ends at the interrupt, so the answer that follows a
                    # decision arrives on the approval response, not the stream.
                    decision_reply = (outcome.approval_body or {}).get("reply")
                    if isinstance(decision_reply, str) and decision_reply.strip():
                        outcome.reply = decision_reply

                    if turn.follow_up is not None:
                        follow_payload = dict(payload)
                        follow_payload["decision"] = turn.follow_up
                        if turn.follow_up_key == "new":
                            follow_payload["idempotency_key"] = (
                                f"{scenario.id}-t{trial}-turn{index}-{uuid.uuid4().hex[:8]}"
                            )
                        follow = await client.post(
                            f"/v1/sessions/{session_id}/approvals",
                            json=follow_payload,
                            headers=headers,
                        )
                        outcome.follow_up_status = follow.status_code
                        outcome.follow_up_body = follow.json()

                _grade_turn(turn, outcome)
                outcomes.append(outcome)

    after_own = await _ticket_ids(factory, scenario.customer)
    after_other = await _ticket_ids(factory, CUSTOMER_B)
    failures = [message for outcome in outcomes for message in outcome.failures]

    new_ids = after_own - before_own
    if len(new_ids) != scenario.tickets_delta:
        failures.append(f"expected {scenario.tickets_delta} new ticket(s), found {len(new_ids)}")
    if after_other != before_other:
        failures.append("another customer's tickets changed")

    if scenario.ticket_subject and new_ids:
        async with factory() as db:
            subjects = await db.scalars(select(Ticket.subject).where(Ticket.id.in_(new_ids)))
            joined = " | ".join(subjects).lower()
        if scenario.ticket_subject.lower() not in joined:
            failures.append(f"new ticket subject missing {scenario.ticket_subject!r}")

    if scenario.ticket_order and new_ids:
        async with factory() as db:
            orders = await db.scalars(select(Ticket.order_id).where(Ticket.id.in_(new_ids)))
            if scenario.ticket_order not in list(orders):
                failures.append(f"new ticket not linked to {scenario.ticket_order}")

    if scenario.pending_status:
        async with factory() as db:
            status_row = await db.scalar(
                select(PendingAction).order_by(PendingAction.created_at.desc()).limit(1)
            )
        actual = status_row.status if status_row else None
        if actual != scenario.pending_status:
            failures.append(
                f"expected latest pending action {scenario.pending_status!r}, got {actual!r}"
            )

    return TrialResult(
        scenario_id=scenario.id,
        category=scenario.category,
        trial=trial,
        passed=not failures,
        turns=outcomes,
        failures=failures,
    )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def config_fingerprint(scenario_file: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    ).stdout.strip()

    import hashlib

    digest = hashlib.sha256(scenario_file.read_bytes()).hexdigest()
    return {
        "commit": head or "unknown",
        "dirty_worktree": bool(dirty),
        "scenario_sha256": digest,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarise(results: list[TrialResult], trials: int) -> dict[str, Any]:
    total_trials = len(results)
    passed_trials = sum(1 for result in results if result.passed)
    by_scenario: dict[str, list[TrialResult]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario_id, []).append(result)

    all_trials_passed = sum(1 for rows in by_scenario.values() if all(row.passed for row in rows))
    latencies = [value for result in results for value in result.latencies]

    categories: dict[str, dict[str, Any]] = {}
    for rows in by_scenario.values():
        category = rows[0].category
        bucket = categories.setdefault(
            category,
            {"scenarios": 0, "all_trials_passed": 0, "trial_successes": 0, "trial_runs": 0},
        )
        bucket["scenarios"] += 1
        bucket["trial_successes"] += sum(1 for row in rows if row.passed)
        bucket["trial_runs"] += len(rows)
        bucket["all_trials_passed"] += int(all(row.passed for row in rows))

    for bucket in categories.values():
        bucket["pass_1"] = bucket["trial_successes"] / bucket["trial_runs"]
        bucket["pass_k"] = bucket["all_trials_passed"] / bucket["scenarios"]

    return {
        "tsr_pass_1": passed_trials / total_trials if total_trials else 0.0,
        "pass_k": all_trials_passed / len(by_scenario) if by_scenario else 0.0,
        "trials_per_scenario": trials,
        "scenarios": len(by_scenario),
        "trial_runs": total_trials,
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "categories": categories,
        "per_scenario": {
            scenario_id: {
                "passed": sum(1 for row in rows if row.passed),
                "trials": len(rows),
            }
            for scenario_id, rows in by_scenario.items()
        },
    }


def render_markdown(
    *,
    summary: dict[str, Any],
    results: list[TrialResult],
    fingerprint: dict[str, Any],
    settings: Settings,
    mode: str,
    generated_at: datetime,
) -> str:
    lines: list[str] = []
    lines.append("# Agent evaluation report")
    lines.append("")
    lines.append(f"Generated: `{generated_at.isoformat(timespec='seconds')}`")
    lines.append(f"Commit: `{fingerprint['commit']}`")
    lines.append(f"Scenario file hash: `{fingerprint['scenario_sha256'][:16]}…`")
    lines.append(f"Mode: `{mode}` · Model: `{settings.llm_model or 'scripted replay'}`")
    if fingerprint["dirty_worktree"]:
        lines.append("")
        lines.append("> **Warning:** the worktree had uncommitted changes during this run.")
        lines.append("> Numbers from this report must not be quoted until the config is committed.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value | Meaning |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Task success rate (`pass^1`) | **{summary['tsr_pass_1']:.1%}** "
        "| share of trial runs that fully succeeded |"
    )
    lines.append(
        f"| `pass^{summary['trials_per_scenario']}` | **{summary['pass_k']:.1%}** "
        "| scenarios that succeeded on every trial |"
    )
    lines.append(f"| Scenarios | {summary['scenarios']} | graded multi-step tasks |")
    lines.append(f"| Trial runs | {summary['trial_runs']} | scenario × trials |")
    lines.append(
        f"| Turn latency p50 / p95 | {summary['latency_ms']['p50']} ms / "
        f"{summary['latency_ms']['p95']} ms | per user message |"
    )
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| Category | Scenarios | `pass^1` | `pass^k` |")
    lines.append("| --- | --- | --- | --- |")
    for name, bucket in sorted(summary["categories"].items()):
        lines.append(
            f"| {name} | {bucket['scenarios']} | {bucket['pass_1']:.1%} | {bucket['pass_k']:.1%} |"
        )
    lines.append("")
    lines.append("## Scenarios")
    lines.append("")
    lines.append("| Scenario | Trials passed | Tools observed (first trial) |")
    lines.append("| --- | --- | --- |")
    seen: set[str] = set()
    for result in results:
        if result.scenario_id in seen:
            continue
        seen.add(result.scenario_id)
        tools = ", ".join(f"`{tool}`" for turn in result.turns for tool in turn.tools) or "—"
        marker = (
            "✅"
            if summary["per_scenario"][result.scenario_id]["passed"]
            == summary["per_scenario"][result.scenario_id]["trials"]
            else "❌"
        )
        row = summary["per_scenario"][result.scenario_id]
        lines.append(
            f"| {marker} `{result.scenario_id}` | {row['passed']}/{row['trials']} | {tools} |"
        )
    lines.append("")

    failures = [result for result in results if not result.passed]
    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("None. Every trial passed every assertion.")
    else:
        for result in failures:
            lines.append(f"### `{result.scenario_id}` (trial {result.trial + 1})")
            for failure in result.failures:
                lines.append(f"- {failure}")
            lines.append("")
    lines.append("")
    lines.append("## What is graded")
    lines.append("")
    lines.append(
        "A trial passes only when all of the following hold: the expected HTTP/error "
        "outcome occurred, the expected tools ran in order and no forbidden tool ran, "
        "the reply contains (and excludes) the required text, any staged write was "
        "confirmed by a human before execution, the ticket delta matches the scenario, "
        "the ticket is linked to the right order, the pending action reached the right "
        "terminal state, and no other customer's rows changed."
    )
    lines.append("")
    if mode == "offline":
        lines.append(
            "Offline mode replays a scripted model plan, so these numbers measure the "
            "deterministic pipeline — routing, guardrails, approval, persistence — not "
            "model quality. Run `--mode live` with a key to measure the model itself."
        )
        lines.append("")
    return "\n".join(lines)
