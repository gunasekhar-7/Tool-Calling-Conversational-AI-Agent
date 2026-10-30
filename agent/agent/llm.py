from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from agent.config import Settings
from agent.errors import ModelUnavailable


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Provider seam. The model id comes from config; nothing is hardcoded here."""
    if not settings.has_model_credentials:
        raise ModelUnavailable("LLM_MODEL and OPENAI_API_KEY must be configured")

    from langchain_openai import ChatOpenAI

    assert settings.openai_api_key is not None
    assert settings.llm_model is not None
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        timeout=60,
        max_retries=1,
    )
