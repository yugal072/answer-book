from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Common interface for all LLM providers.

    Answer generation should depend on this interface,
    not directly on Gemini/OpenAI/etc.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError