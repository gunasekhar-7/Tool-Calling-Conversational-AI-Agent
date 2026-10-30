from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.config import Settings


def test_missing_database_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, demo_api_key="secret")


def test_non_postgres_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="sqlite:///x.db", demo_api_key="secret")


def test_blank_checkpoint_dsn_falls_back_to_database_url() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost:5432/db",
        checkpoint_dsn="",
        demo_api_key="secret",
    )
    assert settings.checkpoint_dsn is None
    assert settings.checkpoint_url == "postgresql://u:p@localhost:5432/db"


def test_checkpoint_url_strips_sqlalchemy_driver() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://u:p@localhost:5432/db",
        demo_api_key="secret",
    )
    assert settings.checkpoint_url.startswith("postgresql://")


def test_blank_model_means_not_configured() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost:5432/db",
        demo_api_key="secret",
        llm_model="  ",
    )
    assert settings.llm_model is None
    assert settings.has_model_credentials is False


def test_credentials_require_both_key_and_model() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost:5432/db",
        demo_api_key="secret",
        llm_model="some-tool-calling-model",
        openai_api_key="sk-test",
    )
    assert settings.has_model_credentials is True
