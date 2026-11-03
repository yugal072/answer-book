"""Provider-neutral application settings.

Each role can select a provider and model independently. Provider credentials and
endpoints are environment configuration, never hard-coded into graph logic.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()

def _float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except (TypeError, ValueError): return default

def _int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except (TypeError, ValueError): return default

def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

class Settings:
    # Provider credentials. Only configure the providers you actually use.
    GROQ_API_KEY = _str("GROQ_API_KEY")
    GOOGLE_API_KEY = _str("GOOGLE_API_KEY", _str("GEMINI_API_KEY"))
    OPENAI_API_KEY = _str("OPENAI_API_KEY")
    OPENAI_COMPATIBLE_API_KEY = _str("OPENAI_COMPATIBLE_API_KEY", _str("DASHSCOPE_API_KEY"))
    OPENAI_COMPATIBLE_BASE_URL = _str("OPENAI_COMPATIBLE_BASE_URL", _str("DASHSCOPE_BASE_URL"))

    # Provider-neutral task routing. Supported provider adapters are documented
    # in app/llm/client.py; unknown providers fail explicitly.
    LLM_DEFAULT_PROVIDER = _str("LLM_DEFAULT_PROVIDER", "groq").lower()
    LLM_CHEAP_PROVIDER = _str("LLM_CHEAP_PROVIDER", LLM_DEFAULT_PROVIDER).lower()
    LLM_STANDARD_PROVIDER = _str("LLM_STANDARD_PROVIDER", LLM_DEFAULT_PROVIDER).lower()
    LLM_REASONING_PROVIDER = _str("LLM_REASONING_PROVIDER", LLM_STANDARD_PROVIDER).lower()
    LLM_PLANNER_PROVIDER = _str("LLM_PLANNER_PROVIDER", LLM_CHEAP_PROVIDER).lower()
    LLM_VERIFIER_PROVIDER = _str("LLM_VERIFIER_PROVIDER", LLM_REASONING_PROVIDER).lower()
    LLM_VISION_PROVIDER = _str("LLM_VISION_PROVIDER", LLM_DEFAULT_PROVIDER).lower()
    LLM_IMAGE_PROVIDER = _str("LLM_IMAGE_PROVIDER", "disabled").lower()

    # Model names are independent of providers. For OpenAI-compatible services,
    # set OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_API_KEY.
    LLM_DEFAULT_MODEL = _str("LLM_DEFAULT_MODEL", _str("GROQ_MODEL", "qwen/qwen3.8-27b"))
    LLM_CHEAP_MODEL = _str("LLM_CHEAP_MODEL", _str("GROQ_CHEAP_MODEL", "llama-3.1-8b-instant"))
    LLM_STANDARD_MODEL = _str("LLM_STANDARD_MODEL", LLM_DEFAULT_MODEL)
    LLM_REASONING_MODEL = _str("LLM_REASONING_MODEL", _str("GROQ_REASONING_MODEL", LLM_DEFAULT_MODEL))
    LLM_PLANNER_MODEL = _str("LLM_PLANNER_MODEL", _str("SOLUTION_PLANNER_MODEL", LLM_CHEAP_MODEL))
    LLM_VERIFIER_MODEL = _str("LLM_VERIFIER_MODEL", LLM_REASONING_MODEL)
    LLM_VISION_MODEL = _str("LLM_VISION_MODEL", LLM_STANDARD_MODEL)
    LLM_IMAGE_MODEL = _str("LLM_IMAGE_MODEL")
    LLM_TEMPERATURE = _float("LLM_TEMPERATURE", _float("GROQ_TEMPERATURE", 0.1))
    LLM_MAX_RETRIES = max(1, _int("LLM_MAX_RETRIES", _int("GROQ_MAX_RETRIES", 3)))
    LLM_RETRY_BASE_SECONDS = max(0.0, _float("LLM_RETRY_BASE_SECONDS", _float("GROQ_RETRY_BASE_SECONDS", 2.0)))

    # Backwards-compatible aliases for older modules, gradually deprecated.
    GROQ_MODEL = LLM_DEFAULT_MODEL
    GROQ_CHEAP_MODEL = LLM_CHEAP_MODEL
    GROQ_REASONING_MODEL = LLM_REASONING_MODEL
    GROQ_TEMPERATURE = LLM_TEMPERATURE
    GROQ_MAX_RETRIES = LLM_MAX_RETRIES
    GROQ_RETRY_BASE_SECONDS = LLM_RETRY_BASE_SECONDS
    SOLUTION_PLANNER_MODEL = LLM_PLANNER_MODEL

    DATABASE_URL = _str("DATABASE_URL")
    SOLUTION_STORE_ENABLED = _str("SOLUTION_STORE_ENABLED", "true").lower() in {"1", "true", "yes"}
    SOLUTION_CACHE_MIN_CONFIDENCE = _float("SOLUTION_CACHE_MIN_CONFIDENCE", 0.90)
    SOLUTION_STORE_MIN_CONFIDENCE = _float("SOLUTION_STORE_MIN_CONFIDENCE", 0.90)
    SOLUTION_MAX_ATTEMPTS = max(1, _int("SOLUTION_MAX_ATTEMPTS", 2))
    SOLUTION_PROMPT_VERSION = _str("SOLUTION_PROMPT_VERSION", "2026-10-v1")

settings = Settings()
