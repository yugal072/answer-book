import os

from dotenv import load_dotenv
from google import genai

from app.llm.base import LLMProvider


load_dotenv()


class GeminiProvider(LLMProvider):
    """
    Gemini implementation of the LLMProvider interface.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text