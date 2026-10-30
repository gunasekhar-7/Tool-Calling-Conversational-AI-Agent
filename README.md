# Order support agent

A support agent that **does things**, not just talks about them: it looks up orders and
tickets in Postgres through typed tools, checks refund policy, and — after a human approves
the exact details — opens a support ticket.

The interesting part is not the model loop. It is everything around it: a policy layer the
model cannot argue with, a durable pause for approval, an atomic hand-off between that pause
and the write, and a graded scenario suite that reports a task success rate instead of a vibe.

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

| Capability | Where it lives |
| --- | --- |
| Agentic tool calling over a real database | `agent/agent/graph.py`, `agent/tools/` |
| Server-side authorization the model cannot forge | `agent/agent/guardrails/policy.py`, session-scoped `customer_id` |
| Human-in-the-loop for every state change | `agent/agent/write_flow.py` (stage → `interrupt()` → atomic consume → execute) |
| Idempotency and replay safety | `agent/db/repositories.py` (`PendingActionRepo.consume`) |
| Guardrails: input screening, argument validation, grounding, iteration cap, rate limit | `agent/agent/guardrails/` |
| Token streaming to a UI | `agent/api/routes/chat.py`, `agent/runtime.py` |
| Measured quality, not claimed quality | `evals/`, `scripts/evaluate.py` |
| Deployment | `Dockerfile`, `docker-compose.yml`, `fly.toml`, `.github/workflows/ci.yml` |

## The approval design (the part most demos get wrong)

A write is never executed inside the model's turn.

1. The model calls `create_ticket`. The graph routes that call to a **write node**, not to a
   tool handler.
2. The write node validates arguments, re-checks that any referenced order belongs to the
   session's customer, and **stages** a `pending_actions` row with a hash of the exact
   arguments. It returns nothing yet.
3. The graph calls `interrupt(preview)`. The HTTP stream ends with a
   `confirmation_required` event carrying the pending id, the preview and its hash.
4. `POST /v1/sessions/{id}/approvals` re-reads the row, compares the preview hash in constant
   time, and **atomically** flips `proposed → executed/rejected` with a conditional `UPDATE`.
   Two concurrent approvals cannot both win; a replay with the same idempotency key returns
   the original ticket instead of creating a second one.
5. Only then does exactly one `INSERT` run, and the graph resumes with a tool message so the
   model can answer with the real ticket id.

A `user_confirmed` flag inside the agent's own state would not be authorization — the model
could set it. Authorization lives outside the model, in a row the model cannot write.

## Guardrails

| Layer | Rule | Failure mode |
| --- | --- | --- |
| Bearer auth | Every `/v1` route | `403 policy_denied` |
| Input screen | 2 000 chars, control chars stripped, override/role-hijack/exfiltration patterns | `403 policy_denied`, model never called |
| Argument validation | Pydantic models, `extra="forbid"`, regex order ids, enumerated priorities | Tool error returned to the model, no side effect |
| Ownership | `customer_id` comes from the session row, never from arguments or the model | `not_found` — identical to "does not exist", so nothing leaks |
| Write policy | One write per turn, no mixed batches, allow-listed tool names | Tool error -> model explains |
| Iteration cap | `MAX_TOOL_CALLS` (default 8) then a terminal "narrow the request" answer | Graceful stop, no loop |
| Grounding | Every `ORD-…`/`TCK-…`/money/date claim is checked against tool receipts; unsupported claims trigger one repair pass, then a safe template | The user never sees an invented status or amount |
| Rate limit | Per-session token bucket | `429` with `Retry-After` |

## Measured results

25 graded multi-step scenarios × 3 trials = **75 trial runs**, all through the real HTTP
surface (SSE + approval endpoint), then graded on trajectory *and* database state.

| Metric | Value |
| --- | --- |
| Task success rate (`pass^1`) | **100.0 %** (75/75) |
| `pass^3` (succeeded on every trial) | **100.0 %** (25/25) |
| Turn latency p50 / p95 | 61.9 ms / 88.5 ms |
| Categories | read 7 · write 3 · multi-step 2 · isolation/safety 6 · guardrails 6 · ambiguity 1 |

A trial only passes when **all** of these hold: the expected HTTP/error outcome occurred, the
expected tools ran in order with no forbidden tool, the reply contains and excludes the right
text, any staged write needed a human decision, the ticket delta matches, the ticket links to
the right order, the pending action reached the right terminal state, and no other customer's
rows changed.

> **Read this honestly:** these numbers are from `--mode offline`, which replays a scripted
> model plan. They measure the deterministic pipeline — routing, policy, approval,
> persistence, grounding — not model quality. `--mode live` (needs `OPENAI_API_KEY` and
> `LLM_MODEL`) measures the model itself and is what a production claim would require.

Reproduce:

```bash
python scripts/evaluate.py --trials 3        # offline, no key needed
python scripts/evaluate.py --mode live --trials 3
```

Reports land in [`evals/report/report.md`](evals/report/report.md) with a commit hash and a
scenario-file hash, so a number always points at the config that produced it. The harness
refuses to look trustworthy twice: a dirty worktree is stamped in the report.

## Run it locally

Requires Python 3.12 and PostgreSQL 16 (or Docker).

```bash
docker compose up -d db                 # Postgres on :55432
uv venv && uv pip install -e ".[dev]"
cp .env.example .env                    # fill DEMO_API_KEY, OPENAI_API_KEY, LLM_MODEL
alembic upgrade head
python scripts/seed.py --reset
python scripts/serve.py                 # API on :8000  (Windows-safe event loop)
streamlit run dashboard/app.py          # UI on :8501
```

Or the whole stack: `docker compose up --build`.

`scripts/serve.py` exists because uvicorn's default Windows loop (Proactor) is incompatible
with psycopg's async mode; containers use plain uvicorn and never take that path.

### API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz`, `/readyz` | Liveness, DB readiness |
| `GET` | `/v1/customers` | Demo customers to sign in as |
| `POST` | `/v1/sessions` | Create a session bound to a customer |
| `POST` | `/v1/sessions/{id}/messages` | SSE stream: `token`, `tool_call`, `confirmation_required`, `grounding_warning`, `replace`, `done`, `error` |
| `POST` | `/v1/sessions/{id}/approvals` | Approve or reject a staged action |
| `GET` | `/v1/sessions/{id}/transcript` | Rehydrate the conversation from checkpoints |

## Deploy

- **Database:** Neon (or any managed Postgres). Set `DATABASE_URL` and, if it differs,
  `CHECKPOINT_DSN`. LangGraph creates its own checkpoint tables on startup.
- **API:** `fly launch --no-deploy --copy-config`, then
  `fly secrets set DATABASE_URL=... CHECKPOINT_DSN=... DEMO_API_KEY=... OPENAI_API_KEY=... LLM_MODEL=...`
  and `fly deploy`. The image runs `alembic upgrade head` before serving, and `/readyz` is the
  health check.
- **Dashboard:** any container host; point `API_BASE` at the API and give it `DEMO_API_KEY`.
  The dashboard holds no database credentials and can do nothing the API would not allow.

Secrets belong in the platform's secret store — never in `fly.toml`, the image, or git.

## Repo map

```
agent/
  api/          FastAPI app, routes, SSE plumbing, error envelope
  agent/        StateGraph, prompts, write flow, guardrails (policy, screen, grounding, claims)
  tools/        Tool registry, read tools, ticket staging, receipts
  db/           SQLAlchemy models, repositories (atomic consume), seed data
migrations/     Alembic
evals/          Scenarios (25), scripted model, grading runner, committed report
dashboard/      Streamlit app + HTTP/SSE client
scripts/        serve.py, seed.py, evaluate.py
tests/          90 unit + integration tests (real Postgres)
docs/superpowers/plans/   Implementation plan
```

## Known limits

- Refund *creation* is out of scope: the agent checks eligibility and can open a ticket, it
  does not move money.
- The demo uses one bearer token for all customers. Real deployments need per-user identity;
  the code already scopes everything to a session's customer, so that swap is local to
  `agent/api/deps.py`.
- Grounding verifies identifiers, amounts and dates. It cannot verify prose, and a
  sufficiently creative sentence with no concrete claims passes by design.
- Offline eval numbers measure the pipeline. The live number is the honest model number, and
  it is not in this repo yet — no key was configured when the report was generated.
