"""Centralized configuration for the application.

Loads environment variables for Groq LLM integration and PostgreSQL database.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv()


class Settings:
    """Application settings and configuration."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")


settings = Settings()
