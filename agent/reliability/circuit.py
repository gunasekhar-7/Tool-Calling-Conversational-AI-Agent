"""Circuit breaker for fast-fail on repeated failures."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

from agent.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation, calls proceed
    OPEN = "open"  # Failing fast, calls rejected
    HALF_OPEN = "half_open"  # Probing recovery with limited probes


@dataclass
class CircuitBreaker:
    """
    Circuit breaker with configurable failure threshold, cooldown, and half-open probes.

    Not thread-safe; intended for single-process async use. Each DB connection pool
    or provider client should have its own instance.
    """

    failure_threshold: int = 5
    cooldown_seconds: int = 30
    half_open_probes: int = 2

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open_successes: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings, prefix: str = "db") -> CircuitBreaker:
        if prefix == "provider":
            return cls(
                failure_threshold=settings.provider_circuit_failure_threshold,
                cooldown_seconds=settings.provider_circuit_cooldown_seconds,
                half_open_probes=settings.provider_circuit_half_open_probes,
            )
        return cls(
            failure_threshold=settings.db_circuit_failure_threshold,
            cooldown_seconds=settings.db_circuit_cooldown_seconds,
            half_open_probes=settings.db_circuit_half_open_probes,
        )

    @property
    def state(self) -> CircuitState:
        self._check_auto_transition()
        return self._state

    def _check_auto_transition(self) -> None:
        """Update internal state if cooldown has passed."""
        if (
            self._state == CircuitState.OPEN
            and time.time() - self._opened_at >= self.cooldown_seconds
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_successes = 0

    async def call(
        self, func: Callable[[], Awaitable[T]], *, operation_name: str = "operation"
    ) -> T:
        """
        Execute func through the circuit breaker.

        Raises CircuitOpen if the circuit is open and not in half-open probe phase.
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                from agent.errors import CircuitOpen

                raise CircuitOpen(f"Circuit breaker open for {operation_name}")

        try:
            result = await func()
            await self._on_success(operation_name)
            return result
        except BaseException as exc:
            await self._on_failure(operation_name, exc)
            raise

    async def _on_success(self, operation_name: str) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_probes:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._half_open_successes = 0
                    logger.info(
                        "circuit_closed_after_probes",
                        extra={"extra_fields": {"operation": operation_name}},
                    )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self, operation_name: str, exc: BaseException) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                self._half_open_successes = 0
                logger.warning(
                    "circuit_reopened_after_half_open_failure",
                    extra={
                        "extra_fields": {
                            "operation": operation_name,
                            "error": type(exc).__name__,
                        }
                    },
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.time()
                    logger.warning(
                        "circuit_opened",
                        extra={
                            "extra_fields": {
                                "operation": operation_name,
                                "failures": self._failure_count,
                                "threshold": self.failure_threshold,
                                "error": type(exc).__name__,
                            }
                        },
                    )

    def force_open(self) -> None:
        """Manually force the circuit open (e.g., on known provider outage)."""
        self._state = CircuitState.OPEN
        self._opened_at = time.time()

    def force_close(self) -> None:
        """Manually force the circuit closed (e.g., after manual recovery)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_successes = 0
