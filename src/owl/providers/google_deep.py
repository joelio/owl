"""Google Gemini Deep Research provider (Interactions API, async polling)."""

from __future__ import annotations

import asyncio
import os
import time

import httpx

from .base import OwlResponse, Provider

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
AGENT_ID = "deep-research-pro-preview-12-2025"
POLL_INTERVAL = 5  # seconds
MAX_POLL_ATTEMPTS = 120  # 10 minutes max


class GoogleDeepProvider(Provider):
    def __init__(self, model_name: str = "gemini-deep-research"):
        self.model_name = model_name

    async def query(self, prompt: str, system_prompt: str | None = None) -> OwlResponse:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return OwlResponse(
                model_name=self.model_name,
                source="google-deep",
                text="",
                error="GOOGLE_API_KEY not set",
            )

        # Prepend system instructions to input for Interactions API
        full_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Start the deep research interaction
                resp = await client.post(
                    f"{GEMINI_BASE_URL}/interactions",
                    params={"key": api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "agent": AGENT_ID,
                        "input": full_input,
                        "background": True,
                        "store": True,
                    },
                )
                resp.raise_for_status()
                interaction = resp.json()
                interaction_name = interaction.get("name", "")

                if not interaction_name:
                    return OwlResponse(
                        model_name=self.model_name,
                        source="google-deep",
                        text="",
                        error="No interaction name returned",
                    )

                # Poll for completion
                for _ in range(MAX_POLL_ATTEMPTS):
                    await asyncio.sleep(POLL_INTERVAL)
                    poll_resp = await client.get(
                        f"{GEMINI_BASE_URL}/{interaction_name}",
                        params={"key": api_key},
                    )
                    poll_resp.raise_for_status()
                    result = poll_resp.json()

                    if result.get("done", False):
                        # Extract the response text
                        output = result.get("response", {})
                        text_parts = []
                        for part in output.get("outputParts", []):
                            if "text" in part:
                                text_parts.append(part["text"])
                        elapsed = time.monotonic() - t0
                        return OwlResponse(
                            model_name=self.model_name,
                            source="google-deep",
                            text="\n".join(text_parts) if text_parts else str(output),
                            elapsed_seconds=round(elapsed, 1),
                        )

                return OwlResponse(
                    model_name=self.model_name,
                    source="google-deep",
                    text="",
                    error="Deep research timed out after polling",
                )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_name,
                source="google-deep",
                text="",
                error=str(e),
            )
