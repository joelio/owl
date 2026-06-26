"""OpenAI Deep Research provider (o3-deep-research, o4-mini-deep-research)."""

from __future__ import annotations

from typing import Any

from .http_base import HttpProvider

OPENAI_BASE_URL = "https://api.openai.com/v1"

# Model name mapping
MODEL_MAP = {
    "o3-deep-research": "o3-deep-research",
    "o4-mini-deep-research": "o4-mini-deep-research",
}


class OpenAIDeepProvider(HttpProvider):
    source = "openai-deep"
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_model = MODEL_MAP.get(model_name, model_name)

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.api_model,
            "input": prompt,
            "tools": [{"type": "web_search"}],
        }
        if system_prompt:
            payload["instructions"] = system_prompt
        return {
            "url": f"{OPENAI_BASE_URL}/responses",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "json": payload,
        }

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        text_parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", ""))
        return {"text": "\n".join(text_parts)}
