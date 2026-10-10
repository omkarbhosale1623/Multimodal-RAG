"""Shared OpenAI client - single client for embedding, LLM, vision, and audio calls."""

from openai import OpenAI

from app.config import settings


def get_openai_client() -> OpenAI:
    """Create an OpenAI client."""
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


# Singleton client — reused across services
openai_client = get_openai_client()
