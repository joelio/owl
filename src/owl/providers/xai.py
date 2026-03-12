"""xAI Grok provider with agentic search and thinking mode."""

from __future__ import annotations

import os
import time

import httpx

from .base import OwlResponse, Provider

XAI_BASE_URL = "https://api.x.ai/v1"


class XAIProvider(Provider):
    def __init__(self, model_name: str = "grok-agentic"):
        self.model_name = model_name

    async def query(self, prompt: str, system_prompt: str | None = None) -> OwlResponse:
        api_key = os.environ.get("XAI_API_KEY", "")
        if not api_key:
            return OwlResponse(
                model_name=self.model_name,
                source="xai",
                text="",
                error="XAI_API_KEY not set",
            )

        try:
            t0 = time.monotonic()
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{XAI_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "grok-4.1-fast",
                        "messages": messages,
                        "tools": [
                            {"type": "web_search"},
                            {"type": "x_search"},
                        ],
                        "chain_limit": 10,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                text = data["choices"][0]["message"]["content"]
                elapsed = time.monotonic() - t0

                return OwlResponse(
                    model_name=self.model_name,
                    source="xai",
                    text=text,
                    elapsed_seconds=round(elapsed, 1),
                )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_name,
                source="xai",
                text="",
                error=str(e),
            )
