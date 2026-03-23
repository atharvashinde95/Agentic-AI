"""
utils/llm_client.py — LangChain-compatible LLM wrapper for Capgemini GenEngine.
Uses langchain-openai's ChatOpenAI with a custom base_url so we don't need
a separate SDK for the Capgemini endpoint.
"""

import logging
from langchain_openai import ChatOpenAI
import config

logger = logging.getLogger(__name__)


def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance pointed at the Capgemini LLM endpoint.
    The Capgemini GenEngine exposes an OpenAI-compatible /v1/chat/completions
    interface, so ChatOpenAI works out-of-the-box.
    """
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS

    logger.info(
        "Initialising LLM | model=%s  base_url=%s  temp=%.2f  max_tokens=%d",
        config.LLM_MODEL, config.LLM_BASE_URL, temp, tokens,
    )

    return ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=temp,
        max_tokens=tokens,
    )
