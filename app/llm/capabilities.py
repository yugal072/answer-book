"""Provider-neutral optional capabilities for image understanding/generation.

Vision is implemented through app.llm.client.interpret_image, so any configured
multimodal provider can be used. Image generation is intentionally a separate
capability: text/vision models do not automatically support image synthesis.
Add an adapter implementing ImageGenerationProvider and register it when a chosen
provider's image-generation API and model are confirmed.
"""
from __future__ import annotations
from typing import Protocol
from app.core.config import settings
from app.llm.client import CapabilityUnavailable, interpret_image

class VisionProvider(Protocol):
    def interpret_figure(self, image: bytes, prompt: str) -> dict: ...

class ImageGenerationProvider(Protocol):
    def generate_diagram(self, prompt: str) -> bytes: ...

class ConfiguredVisionProvider:
    """Uses the configured generic multimodal chat provider/model."""
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or settings.LLM_VISION_PROVIDER
        self.model = model or settings.LLM_VISION_MODEL
    def interpret_figure(self, image: bytes, prompt: str) -> dict:
        return interpret_image(image, prompt, provider=self.provider, model=self.model)

class ConfiguredImageGenerationProvider:
    """Explicit placeholder until an image-generation backend is configured.

    A general text or vision endpoint cannot be assumed to generate image bytes.
    This interface avoids coupling graph code to any single vendor's image API.
    """
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or settings.LLM_IMAGE_PROVIDER).lower()
        self.model = model or settings.LLM_IMAGE_MODEL
    def generate_diagram(self, prompt: str) -> bytes:
        raise CapabilityUnavailable(
            f"Image generation adapter is not registered for provider '{self.provider}'. "
            "Configure a provider-specific ImageGenerationProvider implementation; "
            "do not substitute a text/vision model and pretend it generated an image."
        )
