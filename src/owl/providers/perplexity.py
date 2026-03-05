"""Perplexity Deep Research provider (sonar-deep-research)."""

from __future__ import annotations

import os

import httpx

from .base import OwlResponse, Provider

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


class PerplexityProvider(Provider):
    def __init__(self, model_name: str = "sonar-deep-research"):
        self.model_name = model_name

    async def query(self, prompt: str) -> OwlResponse:
        api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        if not api_key:
            return OwlResponse(
                model_name=self.model_name,
                source="perplexity",
                text="",
                error="PERPLEXITY_API_KEY not set",
            )

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{PERPLEXITY_BASE_URL}/chat/completions",
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
                citations = data.get("citations", [])

                return OwlResponse(
                    model_name=self.model_name,
                    source="perplexity",
                    text=text,
                    citations=citations if citations else None,
                )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_name,
                source="perplexity",
                text="",
                error=str(e),
            )
