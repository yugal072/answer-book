"""Replaceable interfaces for vision understanding and image generation."""

from __future__ import annotations

import base64
import json
from typing import Any, Protocol


class VisionProvider(Protocol):
    """Protocol for providers that extract structured understanding from images."""

    def interpret_figure(self, image: bytes, prompt: str) -> dict[str, Any]:
        """Analyze an image and return structured interpretation data."""
        ...


class ImageGenerationProvider(Protocol):
    """Protocol for providers that generate visual diagrams."""

    def generate_diagram(self, prompt: str) -> bytes:
        """Generate diagram image bytes from a text prompt."""
        ...


class CapabilityUnavailable(RuntimeError):
    """Raised when an optional AI capability (vision or image gen) is requested but unconfigured."""


class OpenAIVisionProvider:
    """Optional OpenAI vision adapter. Requires `openai` and an actual image payload."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise CapabilityUnavailable("OPENAI_API_KEY is not configured for vision.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CapabilityUnavailable(
                "Install the optional `openai` package to enable vision."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def interpret_figure(self, image: bytes, prompt: str) -> dict[str, Any]:
        """Send image bytes to OpenAI Vision model and parse JSON response."""
        data_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")


class OpenAIImageGenerationProvider:
    """Optional image-generation adapter; generated images are artifacts, not proof."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise CapabilityUnavailable(
                "OPENAI_API_KEY is not configured for image generation."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CapabilityUnavailable(
                "Install the optional `openai` package to enable image generation."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate_diagram(self, prompt: str) -> bytes:
        """Generate diagram using OpenAI image generation API."""
        result = self._client.images.generate(
            model=self._model,
            prompt=prompt,
            size="1024x1024",
        )
        item = result.data[0]
        if not getattr(item, "b64_json", None):
            raise CapabilityUnavailable(
                "Image provider did not return image bytes in b64_json form."
            )
        return base64.b64decode(item.b64_json)
