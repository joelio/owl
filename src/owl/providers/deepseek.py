"""DeepSeek Reasoner provider."""

from __future__ import annotations

import os
import time

import httpx

from .base import OwlResponse, Provider
from .retry import with_retry

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(Provider):
    def __init__(self, model_name: str = "deepseek-reasoner"):
        self.model_name = model_name

    async def query(self, prompt: str, system_prompt: str | None = None) -> OwlResponse:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return OwlResponse(
                model_name=self.model_name,
                source="deepseek",
                text="",
                error="DEEPSEEK_API_KEY not set",
            )

        try:
            t0 = time.monotonic()
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient(timeout=300.0) as client:

                async def _do_request():
                    r = await client.post(
                        f"{DEEPSEEK_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model_name,
                            "messages": messages,
                        },
                    )
                    r.raise_for_status()
                    return r

                resp = await with_retry(_do_request)
                data = resp.json()

                text = data["choices"][0]["message"]["content"]
                reasoning = data["choices"][0]["message"].get("reasoning_content")
                elapsed = time.monotonic() - t0

                return OwlResponse(
                    model_name=self.model_name,
                    source="deepseek",
                    text=text,
                    reasoning=reasoning,
                    elapsed_seconds=round(elapsed, 1),
                )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_name,
                source="deepseek",
                text="",
                error=str(e),
            )
