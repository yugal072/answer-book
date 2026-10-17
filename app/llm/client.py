"""Small Groq JSON client with model overrides and bounded retries."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.core.config import settings


class LLMCallError(RuntimeError):
    """A provider call failed or returned invalid structured output."""


def clean_json_text(text: str) -> str:
    """Extract raw JSON text from markdown code blocks if present."""
    text = text.strip()
    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else text


def get_llm_client(model: str | None = None, temperature: float | None = None):
    """Construct a LangChain ChatGroq instance configured for JSON mode."""
    if not settings.GROQ_API_KEY:
        raise LLMCallError("GROQ_API_KEY is not configured; cannot call the LLM.")

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise LLMCallError("Install langchain-groq to enable Groq calls.") from exc

    kwargs: dict[str, Any] = {
        "api_key": settings.GROQ_API_KEY,
        "model": model or settings.GROQ_MODEL,
        "temperature": (
            settings.GROQ_TEMPERATURE if temperature is None else temperature
        ),
    }
    # JSON mode is supported by many Groq models but not necessarily all models.
    kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatGroq(**kwargs)


def call_groq_json(
    system_prompt: str,
    user_prompt: str,
    max_retries: int | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Execute a structured JSON prompt with bounded retries and exponential backoff."""
    retries = max_retries or settings.GROQ_MAX_RETRIES

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as exc:
        raise LLMCallError("Install langchain-core to enable Groq calls.") from exc

    client = get_llm_client(model=model, temperature=temperature)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.invoke(messages)
            payload = json.loads(clean_json_text(str(response.content)))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object, received a different JSON type.")
            return payload
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "429",
                    "rate_limit",
                    "timeout",
                    "temporar",
                    "502",
                    "503",
                    "504",
                )
            )
            if attempt + 1 >= retries or not retryable:
                break

            wait = settings.GROQ_RETRY_BASE_SECONDS * (attempt + 1)
            match = re.search(r"try again in ([0-9.]+)s", str(exc), re.IGNORECASE)
            if match:
                wait = min(float(match.group(1)) + 0.5, 60.0)
            time.sleep(wait)

    raise LLMCallError(
        f"Groq JSON request failed after {attempt + 1} attempt(s): {last_error}"
    ) from last_error
