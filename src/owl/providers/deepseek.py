"""DeepSeek Reasoner provider."""

from __future__ import annotations

from typing import Any

from .http_base import HttpProvider, build_messages, parse_chat_message

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(HttpProvider):
    source = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"

    def __init__(self, model_name: str = "deepseek-reasoner"):
        super().__init__(model_name)

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        return {
            "url": f"{DEEPSEEK_BASE_URL}/chat/completions",
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
        return {"text": message["content"], "reasoning": message.get("reasoning_content")}
