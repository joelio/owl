"""DeepSeek Reasoner provider."""

from __future__ import annotations

import os

import httpx

from .base import OwlResponse, Provider

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(Provider):
    def __init__(self, model_name: str = "deepseek-reasoner"):
        self.model_name = model_name

    async def query(self, prompt: str) -> OwlResponse:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return OwlResponse(
                model_name=self.model_name,
                source="deepseek",
                text="",
                error="DEEPSEEK_API_KEY not set",
            )

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{DEEPSEEK_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                text = data["choices"][0]["message"]["content"]
                reasoning = data["choices"][0]["message"].get("reasoning_content")

                return OwlResponse(
                    model_name=self.model_name,
                    source="deepseek",
                    text=text,
                    reasoning=reasoning,
                )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_name,
                source="deepseek",
                text="",
                error=str(e),
            )
