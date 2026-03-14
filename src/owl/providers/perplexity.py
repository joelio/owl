"""Perplexity Deep Research provider (sonar-deep-research)."""

from __future__ import annotations

import os
import time

import httpx

from .base import OwlResponse, Provider

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


class PerplexityProvider(Provider):
    def __init__(self, model_name: str = "sonar-deep-research"):
        self.model_name = model_name

    async def query(self, prompt: str, system_prompt: str | None = None) -> OwlResponse:
        api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        if not api_key:
            return OwlResponse(
                model_name=self.model_name,
                source="perplexity",
                text="",
                error="PERPLEXITY_API_KEY not set",
            )

        try:
            t0 = time.monotonic()
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{PERPLEXITY_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name,
                        "messages": messages,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                text = data["choices"][0]["message"]["content"]
                citations = data.get("citations", [])
                elapsed = time.monotonic() - t0

                return OwlResponse(
                    model_name=self.model_name,
                    source="perplexity",
                    text=text,
                    citations=citations if citations else None,
                    elapsed_seconds=round(elapsed, 1),
                )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_name,
                source="perplexity",
                text="",
                error=str(e),
            )
