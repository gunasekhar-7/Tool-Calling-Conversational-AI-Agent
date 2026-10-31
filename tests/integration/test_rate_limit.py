from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent.agent.guardrails.rate_limit import RateLimiter
from agent.db.seed import CUSTOMER_A

pytestmark = pytest.mark.integration

START = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)


async def test_burst_is_finite_and_refuses_the_next_call(db) -> None:
    limiter = RateLimiter(rate_per_min=30, burst=2)
    assert (await limiter.check(db, CUSTOMER_A, now=START))[0] is True
    assert (await limiter.check(db, CUSTOMER_A, now=START))[0] is True
    allowed, retry_after = await limiter.check(db, CUSTOMER_A, now=START)
    assert allowed is False
    assert retry_after >= 1


async def test_bucket_refills_after_a_minute(db) -> None:
    limiter = RateLimiter(rate_per_min=30, burst=1)
    assert (await limiter.check(db, CUSTOMER_A, now=START))[0] is True
    assert (await limiter.check(db, CUSTOMER_A, now=START))[0] is False
    later = START + timedelta(seconds=61)
    assert (await limiter.check(db, CUSTOMER_A, now=later))[0] is True


async def test_two_limiter_instances_share_state(db) -> None:
    """Two API workers are two objects over one row, not two private budgets."""
    first = RateLimiter(rate_per_min=30, burst=1)
    second = RateLimiter(rate_per_min=30, burst=1)
    assert (await first.check(db, CUSTOMER_A, now=START))[0] is True
    assert (await second.check(db, CUSTOMER_A, now=START))[0] is False


async def test_retry_after_shrinks_as_tokens_refill(db) -> None:
    # One token per minute: a full refill is 60s, half a refill is 30s.
    limiter = RateLimiter(rate_per_min=1, burst=1)
    assert (await limiter.check(db, CUSTOMER_A, now=START))[0] is True
    _, cold = await limiter.check(db, CUSTOMER_A, now=START)
    _, warm = await limiter.check(db, CUSTOMER_A, now=START + timedelta(seconds=30))
    assert cold == 60
    assert warm == 30


async def test_bucket_table_does_not_grow_per_request(db) -> None:
    from sqlalchemy import func, select

    from agent.db.models import RateLimitBucket

    limiter = RateLimiter(rate_per_min=6000, burst=10)
    for _ in range(5):
        await limiter.check(db, CUSTOMER_A, now=START)
    count = await db.scalar(select(func.count()).select_from(RateLimitBucket))
    assert count == 1
