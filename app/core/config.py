"""Central application settings; environment variables are read once at import."""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    """Parse float from environment variable with fallback."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    """Parse int from environment variable with fallback."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Settings:
    """Runtime configuration values sourced from environment variables."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GROQ_CHEAP_MODEL: str = os.getenv("GROQ_CHEAP_MODEL", "llama-3.1-8b-instant")
    GROQ_REASONING_MODEL: str = os.getenv("GROQ_REASONING_MODEL", "qwen/qwen3.8-27b")
    GROQ_TEMPERATURE: float = _float("GROQ_TEMPERATURE", 0.1)
    GROQ_MAX_RETRIES: int = max(1, _int("GROQ_MAX_RETRIES", 3))
    GROQ_RETRY_BASE_SECONDS: float = max(0.0, _float("GROQ_RETRY_BASE_SECONDS", 2.0))

    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    SOLUTION_STORE_ENABLED: bool = os.getenv(
        "SOLUTION_STORE_ENABLED", "true"
    ).lower() in {"1", "true", "yes"}
    SOLUTION_CACHE_MIN_CONFIDENCE: float = _float(
        "SOLUTION_CACHE_MIN_CONFIDENCE", 0.90
    )
    SOLUTION_STORE_MIN_CONFIDENCE: float = _float(
        "SOLUTION_STORE_MIN_CONFIDENCE", 0.90
    )
    SOLUTION_MAX_ATTEMPTS: int = max(1, _int("SOLUTION_MAX_ATTEMPTS", 2))
    SOLUTION_PROMPT_VERSION: str = os.getenv(
        "SOLUTION_PROMPT_VERSION", "2026-10-v1"
    )
    SOLUTION_PLANNER_MODEL: str = os.getenv("SOLUTION_PLANNER_MODEL", "")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_VISION_MODEL: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
    OPENAI_IMAGE_MODEL: str = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")


settings = Settings()
