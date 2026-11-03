# Provider-Independent Model Interfaces

## Goal
AnswerBook is intended to use multiple providers, including Groq, Google Gemini, Qwen-hosting services, and other compatible APIs. The solver workflow should not depend directly on any vendor SDK.

**Status:** This is the target design. The existing implementation has not yet been replaced by this design.

## Provider versus model
- **Provider:** the service/API used to access a model.
- **Model ID:** the exact identifier accepted by that provider.
- **Capability:** what the model/API can do, such as text generation, image input, structured JSON output, or image generation.
- **Task policy:** which capability and quality/cost tier a workflow task requires.

A model's name alone must not be treated as proof that it supports a capability.

## Suggested application interfaces
```python
from typing import Any, Protocol

class TextModel(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str,
                 model: str, temperature: float = 0.0) -> str: ...

class StructuredTextModel(Protocol):
    def generate_json(self, *, system_prompt: str, user_prompt: str,
                      model: str, schema: dict[str, Any] | None = None) -> dict[str, Any]: ...

class VisionModel(Protocol):
    def analyze_image(self, *, prompt: str, image: bytes,
                      mime_type: str, model: str) -> str: ...

class ImageGenerationModel(Protocol):
    def generate_image(self, *, prompt: str, model: str) -> bytes: ...
```

These signatures are illustrative contracts. Adapt them to the project's sync/async conventions and provider APIs. Do not require every adapter to implement capabilities it does not support.

## Adapter responsibilities
Each adapter should:
- accept a normalized request;
- translate it into the provider's request format;
- normalize response text and structured output;
- expose provider/model metadata;
- report unsupported capabilities clearly;
- classify retryable versus non-retryable errors;
- apply timeouts and bounded retries;
- avoid logging API keys or sensitive request payloads.

Use a registry/factory to resolve configured providers and models. Graph nodes should call the application interface, not instantiate vendor clients.

## Configuration example
```dotenv
LLM_PLANNER_PROVIDER=groq
LLM_PLANNER_MODEL=<provider-supported-model-id>

LLM_SOLVER_PROVIDER=groq
LLM_SOLVER_MODEL=<provider-supported-model-id>

LLM_VERIFIER_PROVIDER=gemini
LLM_VERIFIER_MODEL=<provider-supported-model-id>

LLM_VISION_PROVIDER=gemini
LLM_VISION_MODEL=<vision-capable-model-id>
```

This illustrates the desired configuration shape; it does not guarantee model availability or free-tier access. Configure credentials separately per provider. Selecting a provider with missing credentials should produce a clear configuration error.

Avoid silently substituting an unrelated model when a provider fails. If fallback is enabled, define it explicitly and record the provider/model actually used.

## Capability handling
Check capabilities before dispatch:
- text generation;
- structured output / JSON mode;
- image input / vision;
- tool calling;
- image generation.

Vision and image generation are not interchangeable. A vision model can interpret a diagram without being able to create one. If actual figure bytes or an image reference are unavailable, `has_figure=True` alone is insufficient.

## Errors and retries
Retries should be bounded and distinguish rate limits/transient errors, timeouts, invalid credentials/model IDs, malformed output, and unsupported capability requests. Use retry-after information when available. Avoid blocking sleeps inside async event loops.

## Model selection policy
The planner may choose a task tier based on complexity, marks, image needs, tools, and cost/latency constraints. It should select only configured, capability-compatible models—not route solely from a fixed mapping such as `mcq -> model A`. Record selection decisions for evaluation.
