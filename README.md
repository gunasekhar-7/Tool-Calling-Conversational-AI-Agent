# Order support agent

A support agent that **does things**, not just talks about them: it looks up orders and
tickets in Postgres through typed tools, checks refund policy, and — after a human approves
the exact details — opens a support ticket.

The interesting part is not the model loop. It is everything around it: a policy layer the
model cannot argue with, a durable pause for approval, one commit per request so "nothing was
saved" is true, and a graded scenario suite that reports a task success rate instead of a vibe.

```
                 ┌──────────────────────────── FastAPI ─────────────────────────────┐
  Streamlit      │  auth → input screen → rate limit → SSE stream                   │
  dashboard  ───▶│     │                                                             │
  (chat +        │     ▼                                                             │
   approval      │  LangGraph StateGraph  ── agent ⇄ tools ──▶ write node            │
   card)     ◀───│     │                         │                │                  │
                 │     │                         │          stage pending_actions    │
                 │     │                         │                │                  │
                 │     │                    read tools       interrupt() ─── pause    │
                 │     │                         │                │                  │
                 │     ▼                         ▼                ▼                  │
                 │  grounding check ────────▶ PostgreSQL ◀── POST /approvals         │
                 │  (receipts, not memory)      orders/tickets/receipts/pending       │
                 └───────────────────────────────────────────────────────────────────┘
```

## What it demonstrates

| Capability | Where it lives | Design note |
| --- | --- | --- |
| Agentic tool calling over a real database | `agent/agent/graph.py`, `agent/tools/` | [ADR-1](docs/decisions.md) |
| Server-side authorization the model cannot forge | `agent/agent/guardrails/policy.py`, session-scoped `customer_id` | [ADR-2](docs/decisions.md) |
| Human-in-the-loop for every state change | `agent/agent/write_flow.py` (stage → `interrupt()` → atomic consume → execute) | [ADR-6](docs/decisions.md) |
| Idempotency and replay safety | `agent/db/repositories.py` (`PendingActionRepo.consume`) | [ADR-6](docs/decisions.md) |
| Grounding against tool receipts | `agent/agent/guardrails/grounding.py` | [ADR-3](docs/decisions.md) |
| Shared, bounded rate limiting | `agent/agent/guardrails/rate_limit.py` | [ADR-5](docs/decisions.md) |
| Measured quality, not claimed quality | `evals/`, `scripts/evaluate.py` | [ADR-4](docs/decisions.md) |
| Metrics, alerts and recovery | `agent/observability.py`, [docs/runbook.md](docs/runbook.md) | — |
| Deployment | `Dockerfile`, `docker-compose.yml`, `fly.toml`, `.github/workflows/ci.yml` | — |

## The approval design (the part most demos get wrong)

A write is never executed inside the model's turn.

1. The model calls `create_ticket`. The graph routes that call to a **write node**, not to a
   tool handler.
2. The write node validates arguments, re-checks that any referenced order belongs to the
   session's customer, and **stages** a `pending_actions` row keyed by the tool-call id, with a
   hash of the exact arguments. It returns nothing yet.
3. The graph calls `interrupt(preview)`. The HTTP stream ends with a
   `confirmation_required` event carrying the pending id, the preview and its hash.
4. `POST /v1/sessions/{id}/approvals` loads the row `SELECT … FOR UPDATE`, re-validates it
   (session scope, terminal state, TTL, then the preview hash compared in constant time) and
   consumes it: `proposed → executed` with the ticket insert happening inside the same
   transaction. A second approval with a **different** idempotency key is a 409; a replay with
   the **same** key returns the original ticket instead of creating a second one. Unique
   constraints on `stage_key` and `consume_key` are the backstop if two requests ever race
   past the lock.
5. Only after that commit does the graph resume with a tool message so the model can answer
   using the real ticket id. The write is durable before the reply is written, and no database
   connection is held while the model produces it.

A `user_confirmed` flag inside the agent's own state would not be authorization — the model
could set it. Authorization lives outside the model, in a row the model cannot write. The
commit is likewise outside the graph: repositories `flush()`, and the approval endpoint commits
exactly once, which is why a failed write can honestly say nothing was saved.

## Guardrails

| Layer | Rule | Failure mode |
| --- | --- | --- |
| Bearer auth | Every `/v1` route **and** `/metrics`; one token per customer, no master key | `403 policy_denied` |
| Input screen | 2 000 chars, control chars stripped, override/role-hijack/exfiltration patterns | `403 policy_denied`, model never called |
| Argument validation | Pydantic models, `extra="forbid"`, regex order ids, enumerated priorities | Tool error returned to the model, no side effect |
| Ownership | `customer_id` comes from the authenticated session row, never from arguments or the model | `not_found` — identical to "does not exist", so nothing leaks |
| Write policy | One write per turn, no mixed batches, allow-listed tool names | Tool error → the model explains |
| Iteration cap | `MAX_TOOL_CALLS` (default 8), then one tool-free answer from what was already gathered | Graceful stop, no loop, no canned apology when an answer was in reach |
| Grounding | Every `ORD-…`/`TCK-…`/money/date claim is checked against tool receipts; unsupported claims trigger one repair pass, then a template built from the receipts | The user never sees an invented status or amount |
| Rate limit | Token bucket per customer, stored in Postgres so every worker shares one budget | `429` with `Retry-After` |

## Measured results

25 graded scenarios × 3 trials = **75 trial runs**, driven through the real HTTP surface (SSE
+ approval endpoint) and graded on trajectory *and* resulting database state.

Committed report: [`evals/report/report.md`](evals/report/report.md).

| Run | Metric | Value | 95% CI | n |
| --- | --- | --- | --- | --- |
| offline (pipeline) | Task success rate `pass^1` | 100.0 % | 95.1 % – 100.0 % | 75 |
| offline (pipeline) | `pass^3` (all trials succeeded) | 100.0 % | 86.7 % – 100.0 % | 25 |
| offline (pipeline) | Turn latency p50 / p95 | 107.0 ms / 126.8 ms | — | 75 |
| live (model) | Task success rate | not run — no provider key configured here | — | 0 |

A trial only passes when **all** of these hold: the expected HTTP/error outcome occurred, the
expected tools ran in order with no forbidden tool, the reply contains and excludes the right
text, any staged write needed a human decision, the ticket delta matches, the ticket links to
the right order, the pending action reached the right terminal state, and no other customer's
rows changed.

> **Read this honestly.** The offline row replays a scripted model plan, so it measures the
> deterministic pipeline — routing, policy, approval boundary, persistence, grounding — and
> says nothing about model quality. The live row calls the configured provider and is the only
> number that would support a model-quality claim; it is empty because no key was available
> when the report was generated. `--mode live` requires `OPENAI_API_KEY` and `LLM_MODEL`, needs
> `--trials 2` or more (a single trial cannot produce `pass^k`), and exits non-zero rather than
> writing a 0 % report when the key is missing.

Latency under concurrency is a separate artefact: [`evals/report/load.md`](evals/report/load.md)
records one client alone versus 20 concurrent sessions with a 200 ms simulated provider wait, so
the provider is not the bottleneck being measured. The wait is a synthetic delay, not a model
quality signal; without it the scripted stub is CPU-bound and hides concurrency.

```bash
python scripts/evaluate.py --trials 3            # offline, no key needed
python scripts/evaluate.py --mode live --trials 3  # needs OPENAI_API_KEY + LLM_MODEL
python scripts/loadtest.py --sessions 20 --messages 5
```

Every report carries the commit, the scenario-file hash, the prompt version, the mode and the
interval, and the harness stamps a dirty worktree in the report rather than letting a number
float free of the config that produced it.

## Verify every claim in five commands

Run from a clean worktree with Postgres reachable (`docker compose up -d db`). Green here is
the definition of "it works".

```bash
uv sync --frozen --extra dev --extra dashboard      # lockfile resolves
uv run --frozen ruff check . && uv run --frozen ruff format --check agent tests evals dashboard scripts
uv run --frozen mypy
uv run --frozen pytest                               # 133 tests, real Postgres, 0 skipped
uv run --frozen python scripts/evaluate.py --trials 3 --fail-under 1.0
```

CI runs the same lint, type, test and evaluation commands on every push
(`.github/workflows/ci.yml`), plus both container builds, against a Postgres service on
port 5432.

## Metrics and operations

`GET /metrics` exposes Prometheus series — `agent_http_requests_total`,
`agent_turn_seconds`, `agent_tool_calls_total`, `agent_approvals_total`,
`agent_guardrail_blocks_total` — behind the same bearer auth as everything else. What each one
means, the alert thresholds, how to clear a stuck `proposed` action, how to rotate a customer
token, how to roll back a deploy, and how to read each report: [docs/runbook.md](docs/runbook.md).

## Run it locally

Requires Python 3.12 and PostgreSQL 16 (or Docker).

```bash
docker compose up -d db                    # Postgres on :55432
uv sync --frozen --extra dev --extra dashboard
cp .env.example .env                       # set CUSTOMER_API_KEYS, and OPENAI_API_KEY + LLM_MODEL
uv run alembic upgrade head
uv run python scripts/seed.py --reset
uv run python scripts/serve.py             # API on :8000 (Windows-safe event loop)
uv run streamlit run dashboard/app.py      # UI on :8501
```

Or the whole stack: `docker compose up --build`.

There is no master key. `CUSTOMER_API_KEYS` is a JSON map of token → customer UUID, one per
account; the dashboard's `CUSTOMER_TOKENS` maps a display label to one of those tokens so the
sidebar shows an account picker. The streamlit app creates a session with no body — identity
comes only from the token — and every request it makes carries that token, so switching
accounts switches what the agent can see.

`scripts/serve.py` exists because uvicorn's default Windows loop (Proactor) is incompatible
with psycopg's async mode; containers use plain uvicorn and never take that path.

### API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz`, `/readyz` | Liveness, DB readiness |
| `GET` | `/metrics` | Prometheus exposition (bearer protected) |
| `GET` | `/v1/customers` | The caller's own customer record; there is no way to enumerate others |
| `POST` | `/v1/sessions` | Create a session bound to the authenticated customer (no body) |
| `GET` | `/v1/sessions/{id}` | Confirm the session exists and belongs to the caller |
| `POST` | `/v1/sessions/{id}/messages` | SSE stream: `token`, `tool_call`, `confirmation_required`, `grounding_warning`, `replace`, `done`, `error` |
| `POST` | `/v1/sessions/{id}/approvals` | Approve or reject a staged action (idempotency key + preview hash) |
| `GET` | `/v1/sessions/{id}/transcript` | Rehydrate the conversation from checkpoints |

Error bodies are `{code, message, request_id}` for every failure, including validation errors,
so a client can branch on `code` and quote `request_id` in a report.

## Deploy

- **Database:** Neon (or any managed Postgres). Set `DATABASE_URL` and, if it differs,
  `CHECKPOINT_DSN`. LangGraph creates its own checkpoint tables on startup; Alembic owns the
  application schema.
- **API:** `fly launch --no-deploy --copy-config`, then
  `fly secrets set DATABASE_URL=... CHECKPOINT_DSN=... CUSTOMER_API_KEYS='{"token":"<uuid>"}' OPENAI_API_KEY=... LLM_MODEL=...`
  and `fly deploy`. The image runs `alembic upgrade head` before serving and `/readyz` is the
  health check.
- **Dashboard:** any container host; point `API_BASE` at the API and give it `CUSTOMER_TOKENS`.
  It holds no database credentials and can do nothing the API would not allow.

Secrets belong in the platform's secret store — never in `fly.toml`, the image, or git.

## Repo map

```
agent/
  api/          FastAPI app, routes, SSE plumbing, error envelope, one-commit-per-request boundary
  agent/        StateGraph, prompts, write flow, guardrails (policy, screen, grounding, rate limit)
  tools/        Tool registry, read tools, ticket staging, receipts
  db/           SQLAlchemy models, repositories (atomic consume), seed data
  observability.py   Prometheus registry and the series the runbook references
migrations/     Alembic (initial schema + rate_limit_buckets)
evals/          Scenarios (25), scripted model, grading runner, committed reports
dashboard/      Streamlit app + HTTP/SSE client
scripts/        serve.py, seed.py, evaluate.py, loadtest.py
tests/          133 unit + integration tests (real Postgres)
docs/           runbook.md, decisions.md, superpowers/plans/
```

## Known limits

- Refund *creation* is out of scope: the agent checks eligibility and can open a ticket; it does
  not move money.
- The offline evaluation measures the pipeline, not the model, and no live number is committed
  because no provider key was configured. That is the honesty gap a reviewer should press on
  first.
- Grounding verifies identifiers, amounts and dates. It cannot verify prose, and a creative
  sentence with no concrete claims passes by design.
- A turn holds no database connection while the model streams; tools and staging borrow one
  only for their own short transaction. `DB_POOL_SIZE + DB_MAX_OVERFLOW` therefore bounds
  concurrent database work, not conversations. The remaining per-turn costs are the graph
  rebuild and provider latency, recorded in `evals/report/load.md`.
- Grounding aside, this is single-tenant-per-token: there is no admin scope, no cross-customer
  query, and no way to enumerate other accounts.
