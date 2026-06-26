"""Google Gemini Deep Research provider (Interactions API, async polling)."""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

from .base import OwlResponse, Provider
from .retry import with_retry

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
AGENT_ID = "deep-research-pro-preview-12-2025"
POLL_INTERVAL = 5  # seconds between polls
MAX_RESEARCH_SECONDS = 600  # overall wall-clock budget for the interaction (10 min)


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
                async def _start_interaction():
                    r = await client.post(
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
                    r.raise_for_status()
                    return r

                resp = await with_retry(_start_interaction)
                interaction = resp.json()
                interaction_name = interaction.get("name", "")

                if not interaction_name:
                    return OwlResponse(
                        model_name=self.model_name,
                        source="google-deep",
                        text="",
                        error="No interaction name returned",
                    )

                async def _poll():
                    r = await client.get(
                        f"{GEMINI_BASE_URL}/{interaction_name}",
                        params={"key": api_key},
                    )
                    r.raise_for_status()
                    return r

                # Poll for completion until the overall wall-clock budget runs out.
                # Bounding on elapsed time (not just attempt count) keeps the
                # deadline honest regardless of per-request latency.
                while time.monotonic() - t0 < MAX_RESEARCH_SECONDS:
                    await asyncio.sleep(POLL_INTERVAL)

                    # Retry transient poll failures so a blip doesn't discard
                    # research that may already be minutes in progress.
                    poll_resp = await with_retry(_poll)
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
                    error=f"Deep research timed out after {MAX_RESEARCH_SECONDS}s",
                )
        except Exception as e:
            logger.exception("google-deep provider (%s) failed", self.model_name)
            return OwlResponse(
                model_name=self.model_name,
                source="google-deep",
                text="",
                error=str(e),
            )
