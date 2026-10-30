from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Per-session limiter. Single-process by design; see docs/runbook.md."""

    rate_per_min: int
    burst: int
    tokens: float = field(init=False)
    updated_at: float = field(init=False)
    _clock: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.burst)
        self.updated_at = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(float(self.burst), self.tokens + elapsed * (self.rate_per_min / 60.0))
        self.updated_at = now

    def try_consume(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        self._refill(now)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def retry_after_seconds(self) -> int:
        deficit = max(0.0, 1.0 - self.tokens)
        return max(1, int(deficit / (self.rate_per_min / 60.0)) + 1)


class RateLimiter:
    def __init__(self, rate_per_min: int, burst: int) -> None:
        self.rate_per_min = rate_per_min
        self.burst = burst
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, key: str) -> tuple[bool, int]:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(rate_per_min=self.rate_per_min, burst=self.burst)
            self._buckets[key] = bucket
        if bucket.try_consume():
            return True, 0
        return False, bucket.retry_after_seconds()
