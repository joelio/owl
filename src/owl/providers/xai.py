"""xAI Grok provider with agentic web and X search."""

from __future__ import annotations

from typing import Any

from .http_base import HttpProvider

XAI_BASE_URL = "https://api.x.ai/v1"

DEFAULT_MODEL = "grok-4.5"

# Map owl's friendly model name to a real xAI model id.  Unknown names pass
# through unchanged so a specific id can be used directly.
MODEL_MAP = {
    "grok-agentic": DEFAULT_MODEL,
}


class XAIProvider(HttpProvider):
    """Queries Grok through the Responses API.

    Chat Completions is documented as a deprecated, legacy endpoint offering
    "function calling only", while server-side tools -- web_search and
    x_search among them -- are native to the Responses API.  The agentic
    search this provider exists for is therefore only available there.
    https://docs.x.ai/developers/model-capabilities/text/comparison
    """

    source = "xai"
    api_key_env = "XAI_API_KEY"

    def __init__(self, model_name: str = "grok-agentic"):
        super().__init__(model_name)
        self.api_model = MODEL_MAP.get(model_name, model_name)

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.api_model,
            "input": [{"role": "user", "content": prompt}],
            "tools": [
                {"type": "web_search"},
                {"type": "x_search"},
            ],
        }
        if system_prompt:
            payload["instructions"] = system_prompt
        return {
            "url": f"{XAI_BASE_URL}/responses",
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
