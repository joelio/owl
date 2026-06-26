"""Perplexity Deep Research provider (sonar-deep-research)."""

from __future__ import annotations

from typing import Any

from .http_base import HttpProvider, build_messages, parse_chat_message

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


class PerplexityProvider(HttpProvider):
    source = "perplexity"
    api_key_env = "PERPLEXITY_API_KEY"

    def __init__(self, model_name: str = "sonar-deep-research"):
        super().__init__(model_name)

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        return {
            "url": f"{PERPLEXITY_BASE_URL}/chat/completions",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "json": {
                "model": self.model_name,
                "messages": build_messages(prompt, system_prompt),
            },
        }

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        message = parse_chat_message(data)
        citations = data.get("citations") or None
        return {"text": message["content"], "citations": citations}
