"""Exponential backoff retry for idempotent operations."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from sqlalchemy.exc import OperationalError

from agent.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 5.0
    jitter_factor: float = 0.1

    @classmethod
    def from_settings(cls, settings: Settings) -> RetryConfig:
        return cls(
            max_attempts=settings.db_retry_max_attempts,
            base_delay_seconds=settings.db_retry_base_delay_seconds,
        )


# Retryable SQLSTATE codes per PostgreSQL docs:
# 40001 - serialization_failure
# 40P01 - deadlock_detected
# 08000 - connection_exception (includes 08003, 08006, 08001, 08004, 08007)
# 57P01 - admin_shutdown (pg_ctl stop -m fast)
_RETRYABLE_SQLSTATES = {
    "40001",  # serialization_failure
    "40P01",  # deadlock_detected
    "08000",  # connection_exception
    "08003",  # connection_does_not_exist
    "08006",  # connection_failure
    "08001",  # sqlclient_unable_to_establish_sqlconnection
    "08004",  # sqlserver_rejected_establishment_of_sqlconnection
    "08007",  # transaction_resolution_unknown
    "57P01",  # admin_shutdown
}


def _is_retryable_error(exc: BaseException) -> bool:
    """Check if an exception is a retryable database error."""
    if isinstance(exc, OperationalError):
        orig = getattr(exc, "orig", None)
        if orig is not None:
            sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
            if sqlstate in _RETRYABLE_SQLSTATES:
                return True
            # psycopg3 uses different attribute names
            diag = getattr(orig, "diag", None)
            if diag is not None and getattr(diag, "sqlstate", None) in _RETRYABLE_SQLSTATES:
                return True
    # Also check for asyncpg / psycopg2 connection errors
    return bool(isinstance(exc, (ConnectionError, TimeoutError)))


async def retry_with_backoff[T](
    func: Callable[[], Awaitable[T]],
    config: RetryConfig,
    *,
    operation_name: str = "operation",
    idempotency_key: str | None = None,
) -> T:
    """
    Execute func with exponential backoff retry on transient failures.

    Args:
        func: Async function to execute.
        config: Retry configuration.
        operation_name: Name for logging.
        idempotency_key: If provided, the operation is assumed idempotent and
            safe to retry. If None, only retry on connection errors (not
            serialization/deadlock) to avoid potential double-execution.
    """
    attempt = 0
    last_exc: BaseException | None = None

    while True:
        attempt += 1
        try:
            result = await func()
            return result
        except BaseException as exc:
            last_exc = exc
            retryable = _is_retryable_error(exc)

            if not retryable:
                logger.debug(
                    "non_retryable_error",
                    extra={
                        "extra_fields": {
                            "operation": operation_name,
                            "error": type(exc).__name__,
                        }
                    },
                )
                raise

            if attempt >= config.max_attempts:
                logger.warning(
                    "retry_exhausted",
                    extra={
                        "extra_fields": {
                            "operation": operation_name,
                            "attempts": attempt,
                            "error": type(exc).__name__,
                        }
                    },
                )
                raise

            # For serialization/deadlock errors, only retry if we have an
            # idempotency key (meaning the operation is safe to repeat).
            sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
            if sqlstate in {"40001", "40P01"} and idempotency_key is None:
                logger.warning(
                    "non_idempotent_serialization_error_not_retried",
                    extra={"extra_fields": {"operation": operation_name, "sqlstate": sqlstate}},
                )
                raise

            delay = min(
                config.base_delay_seconds * (2 ** (attempt - 1)),
                config.max_delay_seconds,
            )
            # Add jitter to prevent thundering herd
            jitter = delay * config.jitter_factor * random.uniform(-1, 1)
            delay = max(0, delay + jitter)

            logger.info(
                "retrying_operation",
                extra={
                    "extra_fields": {
                        "operation": operation_name,
                        "attempt": attempt,
                        "max_attempts": config.max_attempts,
                        "delay_seconds": round(delay, 3),
                        "error": type(exc).__name__,
                    }
                },
            )
            await asyncio.sleep(delay)

    # Unreachable
    raise last_exc
