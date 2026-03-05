"""OpenAI Deep Research provider (o3-deep-research, o4-mini-deep-research)."""

from __future__ import annotations

import os

import httpx

from .base import OwlResponse, Provider

OPENAI_BASE_URL = "https://api.openai.com/v1"

# Model name mapping
MODEL_MAP = {
    "o3-deep-research": "o3-deep-research",
    "o4-mini-deep-research": "o4-mini-deep-research",
}


class OpenAIDeepProvider(Provider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_model = MODEL_MAP.get(model_name, model_name)

    async def query(self, prompt: str) -> OwlResponse:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return OwlResponse(
                model_name=self.model_name,
                source="openai-deep",
                text="",
                error="OPENAI_API_KEY not set",
            )

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{OPENAI_BASE_URL}/responses",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.api_model,
                        "input": prompt,
                        "tools": [{"type": "web_search"}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                # Extract text from response output
                text_parts = []
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                text_parts.append(content.get("text", ""))

                return OwlResponse(
                    model_name=self.model_name,
                    source="openai-deep",
                    text="\n".join(text_parts),
                )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_name,
                source="openai-deep",
                text="",
                error=str(e),
            )
