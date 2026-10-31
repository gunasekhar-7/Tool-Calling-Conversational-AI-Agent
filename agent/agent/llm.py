from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from agent.config import Settings
from agent.errors import ModelUnavailable
from agent.tracing import configure_tracing


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Provider seam. The model id comes from config; nothing is hardcoded here."""
    if not settings.has_model_credentials:
        raise ModelUnavailable("LLM_MODEL and OPENAI_API_KEY must be configured")

    # Respan instruments the provider SDK in place, so it has to run before the
    # client exists. A no-op when RESPAN_API_KEY is not set.
    configure_tracing(settings)

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
