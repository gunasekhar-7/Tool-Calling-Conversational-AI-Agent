# Tool-Calling Conversational AI Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `executing-plans` to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A support agent that resolves real order and ticket requests by calling
Postgres-backed tools, pauses for explicit human approval before any write, reports a
measured task success rate over 25 multi-step scenarios, and ships behind a live URL a
recruiter can click.

**Architecture:** FastAPI owns identity, guardrails, persistence and SSE. A LangGraph
`StateGraph` runs the model↔tool loop with a Postgres checkpointer. A policy layer *outside*
the model authorizes every tool call against the server-side `customer_id`. Write tools only
stage a `pending_actions` row; the graph then calls `interrupt()`, and a separate approval
endpoint atomically consumes that row before exactly one idempotent write executes. Streamlit
renders the token stream and the approval card.

**Tech stack:** Python 3.12, FastAPI, LangGraph 1.2.x, `langgraph-checkpoint-postgres`,
`langchain-openai`, SQLAlchemy 2 async + Alembic, PostgreSQL 16, Streamlit,
pytest/pytest-asyncio, ruff + mypy, Docker, Fly.io + Neon.

---

## Global Constraints

- Python 3.12. Pin `langgraph>=1.2,<2`, `langchain-core>=1.0`, `langgraph-checkpoint-postgres`,
  `psycopg[binary,pool]`.
- **No `create_react_agent`** — deprecated in LangGraph v1 in favour of
  `langchain.agents.create_agent`; this plan uses an explicit `StateGraph` because the approval
  node and per-tool policy must be inspectable and testable.
- No experimental APIs on the critical path: do **not** build on `stream_events(version="v3")`.
  Use `astream(..., stream_mode="messages", version="v2")` (stable since 1.1).
- `LLM_MODEL` and `LLM_TEMPERATURE=0` come from env. No model identifier is hardcoded anywhere.
- `customer_id` is never read from model output or request body. It comes from the `sessions`
  row and is injected into tool handlers by the server.
- Every write: **stage → interrupt → atomic consume → idempotent execute**. A `user_confirmed`
  flag inside the agent's own state is not authorization.
- All timestamps UTC. Money as integer minor units (`total_cents`). Logs are JSON with
  `request_id`, `session_id`, `tool`, `latency_ms`.
- No secrets in the repo. `.env.example` lists key names only. Public API requires
  `Authorization: Bearer $DEMO_API_KEY`.
- Reported eval numbers are only valid for a committed config hash (`git rev-parse HEAD` +
  scenario file hash). No metric may be quoted from an uncommitted run.
- Every failure path returns a structured error with a `request_id`; nothing returns a bare
  stack trace.

## Decisions (and what each one costs)

| Decision | Choice | Why | Cost |
| --- | --- | --- | --- |
| Orchestration | LangGraph `StateGraph` over raw function calling | Durable interrupts and checkpointed state are the hard part; raw tool calling means rebuilding a state machine by hand | Framework churn |
| Prebuilt vs custom graph | Custom (`bind_tools` + `ToolNode` + conditional edges) | Approval node, iteration cap and policy gate must be explicit | ~120 lines you own |
| Provider | OpenAI through `build_chat_model` | One seam to swap | Vendor coupling in `llm.py` only |
| Write execution | Separate approval endpoint, not a resume-only UI call | Authority must live outside the paused graph | Two round trips per write |
| Idempotency | `stage_key` + `consume_key` + `preview_hash` | Double-click and replay must not create two tickets | Three columns to keep coherent |
| Grounding | Receipt-based claim verification, not prompt instructions | Prompt instructions fail silently | Only checks identifiers, money, dates |
| Eval harness | Drives the real HTTP surface with a scripted model | Deterministic pipeline regressions, no key needed in CI | Offline numbers are not model numbers |
| Dashboard | Streamlit, plain HTTP client module | Fastest credible UI; client is testable without Streamlit | No component-level UI tests |

---

## Phase 1 — Foundations, data model, configuration

**Files:** `pyproject.toml`, `agent/config.py`, `agent/errors.py`, `agent/logging.py`,
`agent/db/models.py`, `agent/db/session.py`, `alembic.ini`, `migrations/`.

- [x] Create the package layout (`agent/api`, `agent/agent`, `agent/tools`, `agent/db`).
- [x] `Settings` via pydantic-settings: `database_url`, `checkpoint_dsn`, `demo_api_key`,
      `max_tool_calls`, `approval_ttl_seconds`, `rate_limit_*`, `llm_model`,
      `llm_temperature`, `openai_api_key` as `SecretStr`.
- [x] Validate that URLs are `postgresql://`/`postgresql+psycopg://` at load time.
- [x] `checkpoint_url` property converts the SQLAlchemy URL into a libpq DSN.
- [x] Error hierarchy mapping to HTTP: `NotFound` 404, `InvalidInput` 422, `Conflict` 409,
      `PolicyDenied` 403, `ApprovalExpired`/`ApprovalMismatch` 409, `RateLimited` 429,
      `ModelUnavailable` 503.
- [x] JSON logs with a `request_id` context variable and `extra_fields` merging.
- [x] Tables: `customers`, `orders`, `order_items`, `tickets`, `sessions`, `tool_receipts`,
      `pending_actions`, `audit_log`.
- [x] `pending_actions` uniqueness on `stage_key`; `status` in
      `proposed|executed|rejected|expired`; `args_hash`, `consume_key`, `result` JSON.
- [x] Alembic initial migration generated and applied.
- [x] Pin the Windows selector event loop policy in `agent/__init__.py` so scripts, tests and
      the API share one loop type (psycopg async refuses Proactor).

**Milestone:** `alembic upgrade head` succeeds against a fresh database.

**Production notes:** configuration is read once and injected; no module-level `Settings()`.
Every table has an explicit primary key and a `created_at` defaulted in Python, not in SQL, so
timestamps stay testable.

**Portfolio framing:** show the schema diagram and one paragraph on why `pending_actions` is a
first-class table rather than an in-memory dictionary.

---

## Phase 2 — Deterministic seed data

**Files:** `agent/db/seed.py`, `scripts/seed.py`.

- [x] Two customers (Ada, Grace) with deliberately disjoint data.
- [x] Six orders spanning `processing`, `shipped`, `delivered` (inside and outside the 30-day
      refund window) and `cancelled`.
- [x] Pre-existing tickets so "ticket created" is measured as a delta, not an absolute count.
- [x] `reset_and_seed` (truncate then insert) and `seed_if_empty` (idempotent bootstrap).
- [x] Money stored as integer cents; display strings rendered at the tool boundary.

**Milestone:** `python scripts/seed.py --reset` prints one line and the DB matches the fixture.

**Production notes:** seeding is a script, never a startup side effect; the API boots against
whatever schema exists.

---

## Phase 3 — Tool registry and read tools

**Files:** `agent/tools/registry.py`, `agent/tools/context.py`, `agent/tools/orders.py`,
`agent/tools/tickets.py`, `agent/tools/receipts.py`, `agent/tools/serialize.py`.

- [x] `ToolSpec(name, description, args_model, kind, handler)` as the single source of truth for
      JSON schema, dispatch and policy.
- [x] `ToolContext(customer_id, session_id, db, request_id)` built by the server per turn.
- [x] `get_order_status`, `list_customer_orders`, `check_refund_eligibility`, `get_ticket`,
      `list_tickets`; all scoped by `customer_id` in the query, not in a post-filter.
- [x] Ownership failures return the same `not_found` payload as a genuinely missing row.
- [x] Every successful read writes a `tool_receipts` row with the facts used by grounding;
      a denial receipts only `{order_id, found: false}`.
- [x] `to_langchain_tools` adapts specs to `StructuredTool` without duplicating schemas.

**Milestone:** integration tests prove Alice cannot read Grace's order or ticket.

**Production notes:** tools never accept a customer identifier. Adding a tool means one
`ToolSpec`; policy, schemas and receipts come along automatically.

---

## Phase 4 — The agentic loop

**Files:** `agent/agent/state.py`, `agent/agent/prompts.py`, `agent/agent/graph.py`,
`agent/agent/llm.py`.

- [x] `AgentState`: `messages` (reducer `add_messages`), `customer_id`, `tool_calls`,
      `limit_reached`, `pending_action_id`.
- [x] System prompt: identity, scope, "never invent status/amounts", "one write per turn",
      "writes need approval", "no other customers' data".
- [x] `agent` node streams the model, accumulates chunks, strips prior system messages.
- [x] `route_after_agent`: no tool calls → `END`; write tool → `write`; cap exceeded → `limit`;
      otherwise `tools`.
- [x] `ToolNode` for read tools with a structured error handler instead of exceptions.
- [x] `limit` node returns a terminal "narrow the request" answer, not an exception.
- [x] `graph_config` sets `thread_id` and a bounded `recursion_limit`.
- [x] Model built from settings; `ModelUnavailable` when credentials are missing, mapped to 503.

**Milestone:** a scripted model can chain `get_order_status → answer` and the state shows both
messages plus one receipt.

**Production notes:** the graph is a pure function of state plus injected context; nothing reads
globals. Tool errors are data the model can recover from, not crashes.

**Portfolio framing:** include the routing diagram and explain the iteration cap as a cost and
latency control, not a safety afterthought.

---

## Phase 5 — Guardrails

**Files:** `agent/agent/guardrails/input_guard.py`, `.../policy.py`, `.../claims.py`,
`.../grounding.py`, `.../rate_limit.py`.

- [x] Input screen: length, control characters, HTML-ish tags, instruction-override patterns,
      role hijack, prompt exfiltration. Ambiguous text passes and is logged.
- [x] Policy: tool allow-list, one write per turn, argument sanity, write-tool kind check.
- [x] Claim extraction for order ids, ticket ids, money and ISO dates.
- [x] Grounding: every claim must appear in the session's receipts; failures emit
      `grounding_warning`, then one repair pass constrained to receipted facts, then a safe
      template. Never ship an unverified number.
- [x] Per-session token bucket with `Retry-After`.
- [x] Rate limiting and screening happen *before* the model is called, so abuse costs nothing.

**Milestone:** `safety_instruction_override` and `safety_role_hijack` return 403 without a model
call; `guardrail_ungrounded_claim` never emits the invented amount.

**Production notes:** each guardrail is independently unit-tested and has a named error code, so
logs and the eval report speak the same vocabulary.

---

## Phase 6 — Approval write flow

**Files:** `agent/agent/write_flow.py`, `agent/tools/tickets_write.py`,
`agent/db/repositories.py`, `agent/api/routes/approvals.py`.

- [x] `create_ticket` registered as a write tool whose handler refuses (`approval_required`);
      routing sends the call to the write node instead.
- [x] Write node validates args, checks order ownership, stages a row keyed by
      `session:tool_call_id`, logs `write_staged`.
- [x] `interrupt(preview)` returns the pending id, preview body and `preview_hash`.
- [x] `PendingActionRepo.consume` is a conditional `UPDATE ... WHERE status='proposed'` plus
      `consume_key` bookkeeping, so one decision wins and replays return the same ticket.
- [x] Rejection writes nothing and returns a tool message the model can explain.
- [x] Approval endpoint checks: session ownership, terminal states, TTL, constant-time preview
      hash comparison, and that an interrupt is actually pending.
- [x] Audit row per decision (`approval.approve` / `approval.reject`).

**Milestone:** approve creates exactly one ticket; double approval conflicts; a rejected action
cannot be approved afterwards; a replay with the same key returns `replayed=true`.

**Production notes:** the hash binds the decision to the exact arguments the human saw. TTL
prevents stale cards from being approved hours later.

**Portfolio framing:** this is the section recruiters remember. Show the sequence diagram and
the conditional `UPDATE`.

---

## Phase 7 — API surface

**Files:** `agent/api/main.py`, `agent/api/routes/*.py`, `agent/api/sse.py`,
`agent/api/deps.py`, `agent/api/errors.py`, `agent/runtime.py`, `agent/schemas.py`.

- [x] `create_app(settings, model=None, checkpointer=None)` for tests and the harness.
- [x] Lifespan builds the engine, session factory, Postgres checkpointer and runtime; closes
      both on shutdown.
- [x] Bearer auth on every `/v1` route with constant-time comparison.
- [x] `POST /v1/sessions`, `GET /v1/sessions/{id}`, `GET /v1/sessions/{id}/transcript`,
      `GET /v1/customers`.
- [x] `POST /v1/sessions/{id}/messages` streams SSE: `token`, `tool_call`,
      `confirmation_required`, `grounding_warning`, `replace`, `done`, `error`, with keepalive
      comments so proxies do not drop idle sockets.
- [x] `POST /v1/sessions/{id}/approvals` returns `{status, ticket_id, replayed, reply}` — the
      reply is essential, because the turn ended at the interrupt.
- [x] `X-Request-ID` accepted or generated, echoed, and present in every error body.
- [x] Domain errors mapped centrally; unexpected errors logged and returned as
      `internal_error` with the request id.

**Milestone:** `curl` of the stream shows tokens, a tool call, and the confirmation event.

**Production notes:** the runtime is the only component that knows about both the graph and the
database, which is why the eval harness can reuse it through HTTP without special cases.

---

## Phase 8 — Dashboard

**Files:** `dashboard/api_client.py`, `dashboard/app.py`, `dashboard/Dockerfile`.

- [x] `AgentAPI` with `health`, `customers`, `create_session`, `stream_message`, `decide`,
      `transcript`; SSE parsed independently of Streamlit and unit-tested.
- [x] Errors mapped to `{code, message, request_id}` with a human hint per code.
- [x] Sidebar: API base, demo key, customer selector, connection status, new conversation.
- [x] Empty state offers three concrete example prompts instead of a blank box.
- [x] Streaming renders tokens live and shows tool chips as calls happen.
- [x] Approval card shows subject, body, linked order, priority, expiry, and states plainly that
      nothing has been written yet. Approve / Reject are the only ways forward; the input is
      disabled while a decision is pending.
- [x] After a decision: created ticket id, or "nothing was written", plus the assistant's reply.

**Milestone:** a full conversation including one approved ticket, driven from the browser.

**Production notes:** the dashboard has no database access and no decision authority; it can
only do what a user could do through the API.

---

## Phase 9 — Evaluation framework

**Files:** `evals/scenarios.py`, `evals/scripted.py`, `evals/runner.py`, `scripts/evaluate.py`.

- [x] 25 scenarios across read (7), write (3), multi-step (2), isolation/safety (6),
      guardrails (6), ambiguity (1). The module asserts the count.
- [x] `ScriptedModel` replays a plan, emits fresh message ids (LangGraph dedupes by id), and can
      stream, so the real runtime path is exercised.
- [x] Runner drives the app through `httpx.ASGITransport` with a real streamed request, follows
      the approval endpoint, and grades:
      HTTP status, error code, ordered tool subsequence, forbidden tools, required/forbidden
      reply text, approval outcome, `replayed` flag, ticket delta, ticket subject, ticket→order
      link, pending-action terminal state, and that the *other* customer's rows are unchanged.
- [x] Database is reseeded per trial; no scenario can pass by inheriting state.
- [x] `summarise` computes `pass^1`, `pass^k`, per-category rates and latency percentiles.
- [x] Report includes commit hash, scenario file hash, dirty-worktree warning, mode, model, and
      a per-scenario table. Offline mode is labelled as pipeline-only.
- [x] `--fail-under` turns the suite into a CI gate.

**Measured result:** 25 scenarios × 3 trials = 75 runs, `pass^1 = 100 %`, `pass^3 = 100 %`,
p50 61.9 ms / p95 88.5 ms (scripted replay, no provider latency).

**Milestone:** `python scripts/evaluate.py --trials 3` exits 0 and writes
`evals/report/report.{md,json}`.

**Production notes:** the harness is the regression test for behaviour, not just exceptions.
Adding a scenario is data, not code.

**Portfolio framing:** link the committed report, state the definition of success, and be the
first to say the offline numbers are pipeline numbers.

---

## Phase 10 — Packaging, CI, deployment, docs

**Files:** `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `fly.toml`,
`Makefile`, `README.md`, `scripts/serve.py`.

- [x] Multi-stage Dockerfile: uv-built venv in the build stage, slim non-root runtime, health
      check on `/healthz`, `alembic upgrade head` before serving.
- [x] Compose stack: Postgres 16, API (migrate + seed + serve), dashboard image.
- [x] CI: ruff, format check, mypy, migrations, pytest against a Postgres service, then the
      graded eval with `--fail-under 1.0` and the report uploaded as an artifact.
- [x] Fly config with `/readyz` health checks, autostop, and secrets set via `fly secrets`.
- [x] `scripts/serve.py` for Windows (selector loop), since psycopg async refuses Proactor.
- [x] README: architecture, approval design, guardrail table, measured results with caveats,
      local run, API reference, deploy, repo map, known limits.

**Milestone:** `docker compose up --build` yields a working chat + approval flow on `:8501`.

**Production notes:** no secret ever reaches an image or `fly.toml`; the database DSN is a
platform secret. The dashboard image contains no credentials and no database driver.

---

## Verification checklist

```bash
ruff check .
ruff format --check agent tests evals dashboard scripts
mypy
pytest                                   # 90 tests, real Postgres
python scripts/evaluate.py --trials 3    # 100 % pass^1, report committed
```

- [x] All of the above pass on the committed tree.
- [x] Report regenerated from the final commit so the quoted numbers carry that hash.

## Out of scope

- Refund execution (money movement), multi-tenant auth, streaming tool-argument partials,
  voice, and a second model provider. Each is a deliberate cut, documented in the README.
