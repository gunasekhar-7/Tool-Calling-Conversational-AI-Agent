from __future__ import annotations


class DomainError(Exception):
    """Base class for errors that map to a client-visible response."""

    code = "domain_error"
    http_status = 400

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class NotFound(DomainError):
    code = "not_found"
    http_status = 404


class InvalidInput(DomainError):
    code = "invalid_input"
    http_status = 422


class Conflict(DomainError):
    code = "conflict"
    http_status = 409


class PolicyDenied(DomainError):
    code = "policy_denied"
    http_status = 403


class ApprovalExpired(DomainError):
    code = "approval_expired"
    http_status = 409


class ApprovalMismatch(DomainError):
    code = "approval_mismatch"
    http_status = 409


class ApprovalPending(DomainError):
    """A new message arrived while a staged write is waiting for a human."""

    code = "approval_pending"
    http_status = 409


class WriteFailed(DomainError):
    """The approved write could not be committed; nothing was created."""

    code = "write_failed"
    http_status = 500


class RateLimited(DomainError):
    code = "rate_limited"
    http_status = 429

    def __init__(self, message: str, *, retry_after: int = 1) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ModelUnavailable(DomainError):
    code = "model_unavailable"
    http_status = 503


class ProviderDegraded(DomainError):
    """The model provider is degraded; read-only mode is active, writes rejected."""

    code = "provider_degraded"
    http_status = 503


class CircuitOpen(DomainError):
    """A circuit breaker is open; calls fail fast instead of waiting."""

    code = "circuit_open"
    http_status = 503


class ConfigError(RuntimeError):
    """Raised when the service is misconfigured, never shown to end users."""
