"""xAI Grok provider with agentic search and thinking mode."""

from __future__ import annotations

from typing import Any

from .http_base import HttpProvider, build_messages, parse_chat_message

XAI_BASE_URL = "https://api.x.ai/v1"

# Map owl's friendly model names to xAI API model ids.
# Unknown names pass through unchanged so real API ids can be used directly.
MODEL_MAP = {
    "grok-agentic": "grok-4.1-fast",
}


class XAIProvider(HttpProvider):
    source = "xai"
    api_key_env = "XAI_API_KEY"

    def __init__(self, model_name: str = "grok-agentic"):
        super().__init__(model_name)
        self.api_model = MODEL_MAP.get(model_name, model_name)

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        return {
            "url": f"{XAI_BASE_URL}/chat/completions",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "json": {
                "model": self.api_model,
                "messages": build_messages(prompt, system_prompt),
                "tools": [
                    {"type": "web_search"},
                    {"type": "x_search"},
                ],
                "chain_limit": 10,
            },
        }

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        message = parse_chat_message(data)
        return {"text": message["content"]}
