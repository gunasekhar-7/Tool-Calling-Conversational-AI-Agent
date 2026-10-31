from __future__ import annotations

import math
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# One statement consumes a token or refuses: refill, cap at burst, then take one.
# The WHERE clause is what makes a refusal a no-op instead of a partial update.
CONSUME = text(
    """
    INSERT INTO rate_limit_buckets (customer_id, tokens, updated_at)
    VALUES (:customer_id, :burst - 1, :now)
    ON CONFLICT (customer_id) DO UPDATE SET
        tokens = LEAST(
            :burst,
            rate_limit_buckets.tokens
            + EXTRACT(EPOCH FROM (:now - rate_limit_buckets.updated_at)) * :refill_per_second
        ) - 1,
        updated_at = :now
    WHERE LEAST(
        :burst,
        rate_limit_buckets.tokens
        + EXTRACT(EPOCH FROM (:now - rate_limit_buckets.updated_at)) * :refill_per_second
    ) >= 1
    RETURNING tokens
    """
)

PEEK = text("SELECT tokens, updated_at FROM rate_limit_buckets WHERE customer_id = :customer_id")


class RateLimiter:
    """Token bucket persisted per customer.

    In Postgres rather than process memory so two API workers share one budget and
    a restart cannot hand out a fresh burst. The table grows with customers, not
    with requests, so it stays bounded without a sweeper job.
    """

    def __init__(self, *, rate_per_min: int, burst: int) -> None:
        self.rate_per_min = rate_per_min
        self.burst = burst

    @property
    def refill_per_second(self) -> float:
        return self.rate_per_min / 60.0

    async def check(
        self,
        db: AsyncSession,
        customer_id: UUID,
        now: datetime | None = None,
    ) -> tuple[bool, int]:
        moment = now or datetime.now(UTC)
        result = await db.execute(
            CONSUME,
            {
                "customer_id": customer_id,
                "burst": self.burst,
                "now": moment,
                "refill_per_second": self.refill_per_second,
            },
        )
        # A token was taken exactly when the statement returned a row; a refusal
        # updates nothing, so the whole consume is one round trip.
        if result.first() is not None:
            return True, 0
        return False, await self._retry_after(db, customer_id, moment)

    async def _retry_after(self, db: AsyncSession, customer_id: UUID, moment: datetime) -> int:
        row = (await db.execute(PEEK, {"customer_id": customer_id})).one_or_none()
        if row is None:
            return 1
        tokens, updated_at = float(row[0]), row[1]
        elapsed = max(0.0, (moment - updated_at).total_seconds())
        available = min(float(self.burst), tokens + elapsed * self.refill_per_second)
        deficit = max(0.0, 1.0 - available)
        return max(1, math.ceil(deficit / self.refill_per_second))
