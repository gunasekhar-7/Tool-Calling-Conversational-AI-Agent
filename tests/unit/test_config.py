from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from agent.config import Settings

CUSTOMER = UUID("11111111-1111-4111-8111-111111111111")
KEYS = {"key-a": CUSTOMER}


def test_missing_database_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, customer_api_keys=KEYS)


def test_non_postgres_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="sqlite:///x.db", customer_api_keys=KEYS)


def test_no_customer_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_url="postgresql://u:p@localhost:5432/db",
            customer_api_keys={},
        )


def test_tokens_map_to_customers() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost:5432/db",
        customer_api_keys=KEYS,
    )
    assert settings.token_for(CUSTOMER) == "key-a"
    assert settings.token_for(UUID("22222222-2222-4222-8222-222222222222")) is None


def test_blank_checkpoint_dsn_falls_back_to_database_url() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost:5432/db",
        checkpoint_dsn="",
        customer_api_keys=KEYS,
    )
    assert settings.checkpoint_dsn is None
    assert settings.checkpoint_url == "postgresql://u:p@localhost:5432/db"


def test_checkpoint_url_strips_sqlalchemy_driver() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost:5432/db",
        customer_api_keys=KEYS,
    )
    assert settings.checkpoint_url.startswith("postgresql://")


def test_blank_model_means_not_configured() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost:5432/db",
        customer_api_keys=KEYS,
        llm_model="  ",
    )
    assert settings.llm_model is None
    assert settings.has_model_credentials is False


def test_credentials_require_both_key_and_model() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost:5432/db",
        customer_api_keys=KEYS,
        llm_model="some-tool-calling-model",
        openai_api_key="sk-test",
    )
    assert settings.has_model_credentials is True


def test_respan_key_is_optional_and_blank_means_off() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost:5432/db",
        customer_api_keys=KEYS,
        respan_api_key="  ",
    )
    assert settings.respan_api_key is None
    assert settings.has_respan_key is False


def test_respan_key_is_present_when_set() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost:5432/db",
        customer_api_keys=KEYS,
        respan_api_key="rspn-test",
    )
    assert settings.has_respan_key is True
    # The secret stays masked in anything loggable.
    assert "rspn-test" not in repr(settings)


def test_customer_keys_load_from_json_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CUSTOMER_API_KEYS",
        '{"token-one": "11111111-1111-4111-8111-111111111111"}',
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    settings = Settings(_env_file=None)
    assert settings.customer_api_keys == {"token-one": CUSTOMER}
