from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Type

import litellm
from litellm import completion
from pydantic import BaseModel, ValidationError


@dataclass
class ModelConfig:
    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 12000

    @property
    def model_name(self) -> str:
        if "/" in self.model:
            return self.model
        if self.provider.lower() == "openai":
            return f"openai/{self.model}"
        if self.provider.lower() == "gemini":
            return f"gemini/{self.model}"
        return self.model


class MultiProviderLLMClient:
    def __init__(self, default_config: ModelConfig, stage_models: dict[str, str] | None = None):
        self.default_config = default_config
        self.stage_models = stage_models or {}
        litellm.drop_params = True
        self._validate_env(default_config.provider)

    def _validate_env(self, provider: str) -> None:
        provider = provider.lower()
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for provider=openai")
        if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is required for provider=gemini")

    def config_for_stage(self, stage_name: str) -> ModelConfig:
        model_override = self.stage_models.get(stage_name)
        if not model_override:
            return self.default_config
        provider = self.default_config.provider
        model = model_override
        if "/" in model_override:
            provider, model = model_override.split("/", 1)
        self._validate_env(provider)
        return ModelConfig(
            provider=provider,
            model=model,
            temperature=self.default_config.temperature,
            max_tokens=self.default_config.max_tokens,
        )

    def generate_structured(
        self,
        *,
        stage_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        config = self.config_for_stage(stage_name)
        completion_kwargs = {
            "model": config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        temperature = self._temperature_for_model(config)
        if temperature is not None:
            completion_kwargs["temperature"] = temperature

        response = completion(
            **completion_kwargs,
        )
        content = response.choices[0].message.content or ""
        payload = self._parse_json(content)
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
                try:
                    return response_model.model_validate(payload[0])
                except ValidationError:
                    pass
            raise ValueError(
                f"Stage {stage_name} returned invalid JSON for {response_model.__name__}: {exc}\nRaw content:\n{content}"
            ) from exc

    def generate_text(
        self,
        *,
        stage_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        config = self.config_for_stage(stage_name)
        completion_kwargs = {
            "model": config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": config.max_tokens,
        }
        temperature = self._temperature_for_model(config)
        if temperature is not None:
            completion_kwargs["temperature"] = temperature
        response = completion(**completion_kwargs)
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_json(text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
            if match:
                text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
            if match:
                return json.loads(match.group(1))
            raise

    @staticmethod
    def _temperature_for_model(config: ModelConfig) -> float | None:
        model_name = config.model_name.lower()
        if "gpt-5" in model_name:
            return None
        return config.temperature
