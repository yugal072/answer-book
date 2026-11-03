"""Provider-neutral structured text and vision calls.

Adapters:
- groq: langchain-groq / GROQ_API_KEY
- gemini or google: langchain-google-genai / GOOGLE_API_KEY
- openai: langchain-openai / OPENAI_API_KEY
- openai_compatible: langchain-openai with configurable base URL (e.g. Qwen/DashScope
  or another OpenAI-compatible inference endpoint)

A model's actual multimodal/JSON capabilities depend on the provider and model chosen.
We fail explicitly instead of silently switching providers or models.
"""
from __future__ import annotations
import json, re, time
from typing import Any
from app.core.config import settings

class LLMCallError(RuntimeError):
    """A provider call failed or returned invalid structured output."""

class CapabilityUnavailable(LLMCallError):
    """The configured provider lacks credentials, dependency, or required capability."""

def clean_json_text(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text

def _credentials(provider: str) -> tuple[str, str | None]:
    if provider == "groq": return settings.GROQ_API_KEY, None
    if provider in {"gemini", "google", "google_genai"}: return settings.GOOGLE_API_KEY, None
    if provider == "openai": return settings.OPENAI_API_KEY, None
    if provider in {"openai_compatible", "compatible", "dashscope", "qwen"}:
        return settings.OPENAI_COMPATIBLE_API_KEY, settings.OPENAI_COMPATIBLE_BASE_URL or None
    raise CapabilityUnavailable(f"Unsupported LLM provider '{provider}'.")

def build_chat_model(provider: str, model: str, *, temperature: float | None = None, json_mode: bool = False):
    provider = provider.lower().strip()
    api_key, base_url = _credentials(provider)
    if not api_key:
        env = {"groq":"GROQ_API_KEY", "gemini":"GOOGLE_API_KEY", "google":"GOOGLE_API_KEY",
               "google_genai":"GOOGLE_API_KEY", "openai":"OPENAI_API_KEY",
               "openai_compatible":"OPENAI_COMPATIBLE_API_KEY"}.get(provider, "provider API key")
        raise CapabilityUnavailable(f"No API key configured for provider '{provider}' ({env}).")
    temp = settings.LLM_TEMPERATURE if temperature is None else temperature
    try:
        if provider == "groq":
            from langchain_groq import ChatGroq
            kwargs: dict[str, Any] = {"api_key": api_key, "model": model, "temperature": temp}
            if json_mode: kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
            return ChatGroq(**kwargs)
        if provider in {"gemini", "google", "google_genai"}:
            from langchain_google_genai import ChatGoogleGenerativeAI
            kwargs = {"google_api_key": api_key, "model": model, "temperature": temp}
            if json_mode: kwargs["response_mime_type"] = "application/json"
            return ChatGoogleGenerativeAI(**kwargs)
        from langchain_openai import ChatOpenAI
        kwargs = {"api_key": api_key, "model": model, "temperature": temp}
        if provider in {"openai_compatible", "compatible", "dashscope", "qwen"}:
            if not base_url:
                raise CapabilityUnavailable("Set OPENAI_COMPATIBLE_BASE_URL for the compatible endpoint (for example, Qwen/DashScope).")
            kwargs["base_url"] = base_url
        if json_mode: kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)
    except ImportError as exc:
        package = {"groq":"langchain-groq", "gemini":"langchain-google-genai", "google":"langchain-google-genai",
                   "google_genai":"langchain-google-genai", "openai":"langchain-openai",
                   "openai_compatible":"langchain-openai", "compatible":"langchain-openai",
                   "dashscope":"langchain-openai", "qwen":"langchain-openai"}.get(provider, "provider integration")
        raise CapabilityUnavailable(f"Install '{package}' to use provider '{provider}'.") from exc

def call_json(system_prompt: str, user_prompt: str, *, provider: str | None = None,
              model: str | None = None, temperature: float | None = None,
              max_retries: int | None = None) -> dict[str, Any]:
    provider = (provider or settings.LLM_DEFAULT_PROVIDER).lower()
    model = model or settings.LLM_DEFAULT_MODEL
    retries = max_retries or settings.LLM_MAX_RETRIES
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as exc:
        raise CapabilityUnavailable("Install langchain-core to use model providers.") from exc
    client = build_chat_model(provider, model, temperature=temperature, json_mode=True)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.invoke(messages)
            content = response.content
            if isinstance(content, list):
                content = "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
            payload = json.loads(clean_json_text(str(content)))
            if not isinstance(payload, dict): raise ValueError("Expected a JSON object.")
            return payload
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = any(token in message for token in ("429", "rate_limit", "timeout", "temporar", "502", "503", "504"))
            if attempt + 1 >= retries or not retryable: break
            wait = settings.LLM_RETRY_BASE_SECONDS * (attempt + 1)
            match = re.search(r"try again in ([0-9.]+)s", str(exc), re.IGNORECASE)
            if match: wait = min(float(match.group(1)) + 0.5, 60.0)
            time.sleep(wait)
    raise LLMCallError(f"JSON request failed for provider={provider}, model={model} after {attempt + 1} attempt(s): {last_error}") from last_error

def call_groq_json(system_prompt: str, user_prompt: str, max_retries: int | None = None,
                   model: str | None = None, temperature: float | None = None) -> dict[str, Any]:
    """Compatibility wrapper; new code should use call_json(provider=..., model=...)."""
    return call_json(system_prompt, user_prompt, provider="groq", model=model or settings.LLM_DEFAULT_MODEL,
                     temperature=temperature, max_retries=max_retries)

def interpret_image(image: bytes, prompt: str, *, provider: str | None = None,
                    model: str | None = None) -> dict[str, Any]:
    """Send image bytes to a configured multimodal chat model and return JSON.

    Uses a standard LangChain image_url message shape. Not every model behind every
    provider supports image input; such incompatibility is surfaced as an error.
    """
    import base64
    from langchain_core.messages import HumanMessage, SystemMessage
    provider = (provider or settings.LLM_VISION_PROVIDER).lower()
    model = model or settings.LLM_VISION_MODEL
    client = build_chat_model(provider, model, json_mode=True)
    data_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    messages = [SystemMessage(content="Analyze the supplied image carefully. Return only a JSON object."),
                HumanMessage(content=[{"type":"text", "text":prompt}, {"type":"image_url", "image_url":{"url":data_url}}])]
    try:
        response = client.invoke(messages)
        content = response.content
        if isinstance(content, list): content = "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
        result = json.loads(clean_json_text(str(content)))
        if not isinstance(result, dict): raise ValueError("Expected JSON object from vision model.")
        return result
    except Exception as exc:
        raise LLMCallError(f"Image interpretation failed for provider={provider}, model={model}: {exc}") from exc
