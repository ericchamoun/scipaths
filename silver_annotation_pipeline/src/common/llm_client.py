import os
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model_name = os.getenv("LLM_MODEL", "gemini-3.1-pro-preview")

        if self.provider == "gemini":
            if genai is None:
                raise ImportError("google-generativeai not installed.")
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY not set.")
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            raise NotImplementedError("Only Gemini provider is wired for now.")

    def call(self, prompt: str, schema: Optional[dict] = None) -> str:
        """
        Call the underlying LLM.

        If `schema` is provided (as a plain JSON schema dict), and provider is Gemini,
        use it as response_schema with JSON mime type.
        """
        if self.provider == "gemini":
            if schema and GenerationConfig is not None:
                config = GenerationConfig(
                    response_schema=schema,
                    response_mime_type="application/json",
                )
                response = self.model.generate_content(
                    prompt,
                    generation_config=config,
                )
            else:
                response = self.model.generate_content(prompt)

            text = getattr(response, "text", "")
            if not text:
                raise RuntimeError("LLM response did not contain text.")
            return text

        raise NotImplementedError("Schema-based calls only wired for Gemini right now.")
