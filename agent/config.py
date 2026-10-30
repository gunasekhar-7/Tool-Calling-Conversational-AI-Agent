from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every value is overridable by environment variable."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: str = "development"
    database_url: str
    checkpoint_dsn: str | None = None
    openai_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.0

    demo_api_key: str = Field(min_length=4)
    max_tool_calls: int = Field(default=8, ge=1, le=32)
    approval_ttl_seconds: int = Field(default=900, ge=1)
    max_message_chars: int = Field(default=2000, ge=1)
    rate_limit_per_min: int = Field(default=30, ge=1)
    rate_limit_burst: int = Field(default=10, ge=1)
    log_level: str = "INFO"
    sse_keepalive_seconds: float = 15.0
    cors_origins: list[str] = []

    @field_validator("checkpoint_dsn", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        return value or None

    @field_validator("llm_model", mode="before")
    @classmethod
    def _blank_model_to_none(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("database_url", "checkpoint_dsn")
    @classmethod
    def _require_postgres(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("must be a postgresql:// or postgresql+psycopg:// URL")
        return value

    @property
    def checkpoint_url(self) -> str:
        """LangGraph's saver wants a libpq DSN, not a SQLAlchemy URL."""
        raw = self.checkpoint_dsn or self.database_url
        return raw.replace("postgresql+psycopg://", "postgresql://")

    @property
    def has_model_credentials(self) -> bool:
        key = self.openai_api_key.get_secret_value().strip() if self.openai_api_key else ""
        return bool(key and self.llm_model)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
