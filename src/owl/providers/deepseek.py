"""DeepSeek thinking-mode provider."""

from __future__ import annotations

from typing import Any

from .http_base import HttpProvider, build_messages, parse_chat_message

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEFAULT_MODEL = "deepseek-v4-flash"

# The legacy names `deepseek-chat` and `deepseek-reasoner` were discontinued
# on 2026-07-24.  `deepseek-reasoner` was the V4-Flash thinking mode, so the
# equivalent today is v4-flash with thinking enabled explicitly.
# https://api-docs.deepseek.com/updates/
LEGACY_MODEL_MAP = {
    "deepseek-reasoner": DEFAULT_MODEL,
    "deepseek-chat": DEFAULT_MODEL,
}


class DeepSeekProvider(HttpProvider):
    source = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__(model_name)
        self.api_model = LEGACY_MODEL_MAP.get(model_name, model_name)

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        return {
            "url": f"{DEEPSEEK_BASE_URL}/chat/completions",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "json": {
                "model": self.api_model,
                "messages": build_messages(prompt, system_prompt),
                # Thinking is opt-in on V4; the old reasoner model had it on
                # implicitly.  Sampling parameters are ignored in this mode.
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            },
        }

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        message = parse_chat_message(data)
        return {"text": message["content"], "reasoning": message.get("reasoning_content")}
