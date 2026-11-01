# Runbook

What to do when this service misbehaves. Every claim here refers to code in this
repository: metric names come from `agent/observability.py`, error codes from
`agent/errors.py`, and configuration from `agent/config.py`.

## 0. Service Level Objectives (SLOs)

| SLO | Target | Measurement Window |
| --- | --- | --- |
| Availability | 99.9% | 30-day rolling |
| Turn latency p95 | < 2 s | 5-min rolling |
| Error rate (5xx) | < 1% | 5-min rolling |
| Approval success rate | > 99% | 1-hour rolling |
| Degraded-mode availability | 100% (read tools) | Continuous |

SLOs are measured via Prometheus metrics (see §2). Alert rules in `monitoring/alerts.yml`
fire when error budgets are consumed.

## 1. Health and where to look first

| Signal | Meaning | Where |
| --- | --- | --- |
| `GET /healthz` | Process is up. Says nothing about the database. | Load balancer liveness |
| `GET /readyz` | `SELECT 1` against Postgres succeeded; includes `model_ready`, `degraded`, `db_circuit`, `provider_circuit`. | Load balancer readiness, Fly health check |
| `GET /metrics` | Prometheus exposition, **bearer protected** | Scrape with a customer token |
| stdout JSON lines | One object per event, with `request_id` | `fly logs`, `docker compose logs api` |

If a turn fails, the SSE stream ends with `event: error` carrying `code`, `message`
and `request_id`, and the same `request_id` appears in the logs — grep for it.

## 2. Metrics and alert thresholds

All series are per process. With more than one machine, sum across instances.

| Series | Labels | What it tells you | Alert |
| --- | --- | --- | --- |
| `agent_http_requests_total` | `route`, `method`, `status` | Traffic and error mix by route. `route="unmatched"` means 404s hitting no handler. | 5xx rate > 1% over 5 min |
| `agent_turn_seconds` | `outcome` = `done`, `awaiting_approval`, `error`, `degraded_done` | Time from message to terminal event, including provider time. | p95 > 15 s for 10 min |
| `agent_tool_calls_total` | `tool`, `outcome` = `ok`, `not_found`, `denied`, `error`, `unavailable`, `staged`, `executed`, `rejected` | Which tools run and how they end. A rising `denied` means the model is proposing actions policy refuses. | `error` > 5% of calls over 15 min |
| `agent_approvals_total` | `decision` = `approve`, `reject` | Human decisions on staged writes. | Approve share collapsing to 0% with staged writes still rising: the preview is confusing people |
| `agent_guardrail_blocks_total` | `reason` | Requests stopped before a model call. `rate_limited` is produced by the limiter, the rest by `screen_input`. `provider_degraded` indicates degraded mode. | Any sustained spike is a signal to read the logs, not to relax the guardrail |
| `agent_degraded_turns_total` | `outcome` | Turns executed in degraded read-only mode. | Sustained > 0 for 5 min: provider is down |
| `agent_circuit_state` | `circuit` = `db`, `provider` | Circuit breaker state (0=closed, 1=half_open, 2=open). | `db` or `provider` = 2 (open) for > 1 min |
| `agent_retry_attempts_total` | `operation`, `result` = `success`, `failure` | Database retry attempts. | Retry rate > 10% of operations |
| `agent_body_size_rejections_total` | — | Requests rejected due to body size limit. | Any sustained spike |

`agent_turn_seconds` is the honest latency number. It includes the provider call.
`evals/report/load.md` is a different measurement (no provider); do not mix them.

Model-call tracing is opt-in. With `RESPAN_API_KEY` set, tracing is initialized once
before the first model client and flushed at shutdown; unset, the SDK is never
imported. It exports prompts and tool payloads to a third party, so treat the key as
a data-processing decision, not just an observability toggle.

## 2a. Alert rules

Alert rules are defined in `monitoring/alerts.yml` and loaded by Prometheus/Alertmanager.
Key alerts:

- **HighErrorRate**: `rate(agent_http_requests_total{status=~"5.."}[5m]) > 0.01`
- **HighLatency**: `histogram_quantile(0.95, rate(agent_turn_seconds_bucket[5m])) > 15`
- **ProviderCircuitOpen**: `agent_circuit_state{circuit="provider"} == 2`
- **DBCircuitOpen**: `agent_circuit_state{circuit="db"} == 2`
- **DegradedModeActive**: `rate(agent_degraded_turns_total[5m]) > 0`
- **HighRetryRate**: `rate(agent_retry_attempts_total{result="failure"}[5m]) / rate(agent_retry_attempts_total[5m]) > 0.1`
- **BodySizeRejections**: `rate(agent_body_size_rejections_total[5m]) > 0`

## 3. Error codes and what a client should do

| HTTP | `code` | Meaning | Client behaviour |
| --- | --- | --- | --- |
| 403 | `policy_denied` | Missing or invalid bearer token (no token is a match in `CUSTOMER_API_KEYS`), or the message tripped the input screen | Do not retry. Show the message; for a token problem, re-authenticate or rotate |
| 409 | `approval_pending` | A staged write is still live. A new message would leave its tool call unanswered | Keep the approval card visible and block new messages |
| 409 | `approval_expired` | The TTL passed before the decision landed. The paused graph is settled as a rejection; no write happened | Re-ask the agent; the session remains valid |
| 409 | `approval_mismatch` | The `preview_hash` does not match the staged arguments | Re-render the preview from a fresh turn; never retry with the old hash |
| 409 | `conflict` | The action was already executed with a different idempotency key, or already rejected | Treat as final. Fetch the ticket to confirm |
| 413 | `request_too_large` | Request body exceeds `MAX_REQUEST_BYTES` (64 KiB), enforced by streaming byte count | Fix the request; this is refused before parsing |
| 422 | `invalid_input` | Empty, too long (`MAX_MESSAGE_CHARS`), or a malformed body | Fix the request |
| 429 | `rate_limited` | Token bucket empty for this customer | Wait `Retry-After` seconds |
| 503 | `model_unavailable` | `LLM_MODEL` / `OPENAI_API_KEY` missing, or no model could be built | Operator action, not a client retry loop |
| 503 | `provider_degraded` | Provider unavailable; read-only mode active. Read tools work, writes rejected. | Retry read operations; writes will fail until provider recovers |
| 503 | `circuit_open` | Circuit breaker open (DB or provider). Calls fail fast. | Back off and retry after cooldown |
| 500 | `write_failed` | The approved write failed and was rolled back. Nothing was created | Safe to retry the approval with a new idempotency key |

The 409 `approval_pending` is deliberate: the graph is paused with an unanswered tool
call, so the API refuses a new message rather than letting it inherit that call
(`AgentRuntime.settle_approval_state`). Once the approval expires, the next message
settles the interrupt as a **silent rejection** before the new turn starts: the write
node records a rejected tool result and routes to END. No model call is made, no
ticket is created, and the provider never sees an assistant tool call without a
matching tool message.

## 4. Degraded mode (provider unavailable)

When the LLM provider is unreachable or the provider circuit breaker opens, the service
enters **degraded read-only mode** (controlled by `DEGRADED_READ_ONLY=true`):

- **Read tools work**: `get_order_status`, `list_customer_orders`, `get_ticket`,
  `check_refund_eligibility` execute normally and return grounded answers.
- **Writes rejected**: `create_ticket` returns `provider_degraded` (503) immediately
  without staging.
- **SSE still streams**: Tokens and tool calls are emitted; the turn ends with `done`
  or `error` (`provider_degraded`).
- **Approval endpoint**: Returns `provider_degraded` for any write approval.
- **Recovery**: Automatic when provider circuit closes (half-open probes succeed).
  No deploy needed.

Runbook entry:
1. Check `/readyz` — `degraded: true`, `provider_circuit: "open"`.
2. Check provider status (OpenAI dashboard, network connectivity).
3. If provider is down, inform stakeholders that write operations are unavailable.
   Read operations remain functional.
4. When provider recovers, circuit half-open probes will close it automatically.
   No manual action required unless `PROVIDER_CIRCUIT_COOLDOWN_SECONDS` needs tuning.

## 5. A proposed action is stuck

A staged row can sit in `proposed` in three ways: the user closed the tab, the TTL
passed, or the approval response was lost.

1. It is already harmless after the TTL: the next approval attempt marks it
   `expired` and executes nothing (`PendingActionRepo.consume`).
2. No manual graph cleanup is needed. The next message closes an expired interrupt
   silently and starts a fresh turn; a row marked `expired` by hand is settled the
   same way.
3. To stop it counting as "open" for a session sooner:

   ```sql
   UPDATE pending_actions
      SET status = 'expired', decided_at = now()
    WHERE status = 'proposed' AND expires_at < now();
   ```

4. If a row is `executed`, the ticket exists — the process died after the commit,
   which is the only way that state is visible. Nothing to repair; the audit row
   `approval.approve` names the ticket. The next message replays the executed outcome
   into the graph so its tool call is answered with that ticket id.

Never "fix" a `proposed` row by setting it to `executed`: that would claim a write
that never happened. Recovery is always expire-and-restage.

## 5. Rotate a customer token

Tokens are configuration, not data (`CUSTOMER_API_KEYS`, a JSON object of
token → customer UUID).

1. Add the new token next to the old one and deploy. Both work; nothing breaks.
2. Move the client (dashboard `CUSTOMER_TOKENS`, the customer's integration) over.
3. Remove the old token and deploy again.

Sessions belong to a customer, not to a token, so a live conversation keeps working
across the rotation. Removing a token with no replacement is the only way to revoke
access; it takes effect on the next deploy and closes that customer's API access
immediately.

## 6. Deploy and roll back

The API container runs `alembic upgrade head` before `uvicorn` starts, so a broken
migration fails the release and the old release keeps serving.

```bash
fly releases                    # find the target version
fly deploy --image <previous-image>
```

Migrations are additive (new tables, new nullable columns) as a rule, which is what
makes a code rollback safe without a schema rollback. The only schema added since
the initial revision is `rate_limit_buckets`; dropping it (`alembic downgrade -1`)
costs nothing but a fresh burst per customer.

Rollback does not undo committed writes. A ticket created by an approved action
stays; cancel it in the data, not in the deployment.

## 7. Capacity

- A turn holds no Postgres connection while the model streams. Each tool, the
  staging step and the approval commit borrow one for their own short transaction,
  so `DB_POOL_SIZE + DB_MAX_OVERFLOW` bounds concurrent database work rather than
  concurrent conversations. Raising it does not make a single query faster.
- Each turn rebuilds the tool-bound graph (tools close over the request's session),
  a fixed cost that does not amortize. `evals/report/load.md` records what that
  costs on the machine that generated it.
- With a real model the dominant cost is provider latency, which overlaps across
  turns; with the scripted stand-in the turn is CPU-bound on one event loop, so the
  load numbers without `--think-ms` understate concurrency, not overstate it.

## 8. Reading an evaluation report

`evals/report/report.md` has two modes, and they measure different things:

- **offline** replays scripted model plans. It measures routing, guardrails, the
  approval boundary and persistence. A number here is a statement about the
  pipeline, never about model quality.
- **live** calls the configured provider. It measures the model on the same 25
  scenarios. `pass^k` needs `--trials 2` or more, which is why the CLI refuses one.

Both carry `n`, a 95% Wilson interval, the model id, the prompt version and the
commit. If the report says the worktree was dirty, the numbers are not quotable —
regenerate on a clean tree.

`evals/report/load.md` is a third, separate artefact: platform latency with the
provider removed. It is not a success rate.

## 9. Known limits

- Single region, single primary database. No read replicas, no failover.
- One Postgres database holds both application data and LangGraph checkpoints. A
  checkpoint write and a ticket write are not atomic across a crash; the design
  instead makes every write idempotent (`stage_key`, `consume_key`) so a retry is
  the recovery path.
- The rate limiter is per customer, not per IP or per endpoint. One customer
  exhausting their budget does not affect another.
- The dashboard is a demo client with no login of its own. It holds one customer
  token and can do nothing the API would not allow, but it must sit behind your auth
  before being exposed.
- With `RESPAN_API_KEY` set, prompts and tool payloads leave the deployment in traces.
- Realtime collaboration is out of scope: sessions belong to one customer at a
  time and there is no websocket fan-out.
