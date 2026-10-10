"""Application configuration — loads and validates all env vars at startup."""

import os
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    # OpenAI API (unified provider for embedding + LLM)
    openai_api_key: str
    openai_base_url: str
    openai_embedding_model: str
    openai_llm_model: str

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str

    # App
    cors_origins: list[str]
    log_level: str


def _require(key: str) -> str:
    """Get a required env var or fail fast."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def load_settings() -> Settings:
    """Load and validate all settings from environment. Fail fast on missing vars."""
    settings = Settings(
        openai_api_key=_require("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        pinecone_api_key=_require("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "rag-multimodal"),
        cors_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,*").split(","),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
    logger.info(
        "Settings loaded: openai_base_url=%s, embedding_model=%s, llm_model=%s, pinecone_index=%s",
        settings.openai_base_url,
        settings.openai_embedding_model,
        settings.openai_llm_model,
        settings.pinecone_index_name,
    )
    return settings


# Singleton — imported across the app
settings = load_settings()
