"""Google Gemini Deep Research provider (Interactions API, async polling)."""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

from .base import OwlResponse, Provider
from .errors import describe_error
from .retry import with_retry

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Newer agents also exist: deep-research-preview-04-2026 and
# deep-research-max-preview-04-2026.  This one remains listed as valid.
# https://ai.google.dev/api/interactions-api
AGENT_ID = "deep-research-pro-preview-12-2025"

POLL_INTERVAL = 5  # seconds between polls
MAX_RESEARCH_SECONDS = 600  # overall wall-clock budget for the interaction (10 min)

# The interaction reports progress through a `status` string, not a `done`
# boolean, and the generated text lives in `steps[].content[].text`.
_FAILED_STATUSES = frozenset({"failed", "cancelled", "incomplete", "budget_exceeded"})


def extract_text(result: dict) -> str:
    """Pull the model output out of a completed interaction."""
    text_parts: list[str] = []
    for step in result.get("steps", []):
        if not isinstance(step, dict):
            continue
        for item in step.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                text_parts.append(str(item["text"]))
    return "\n".join(text_parts)


def _failure_reason(result: dict) -> str:
    """Best-effort reason for an interaction that did not complete."""
    error = result.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return "no reason given"


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
            # Authenticate with the x-goog-api-key header rather than a ?key=
            # query parameter.  Google recommends the header precisely because
            # a key in the URL leaks into logs and error messages, and httpx
            # embeds the full URL in str(HTTPStatusError).
            # https://ai.google.dev/gemini-api/docs/api-key
            async with httpx.AsyncClient(
                timeout=30.0, headers={"x-goog-api-key": api_key}
            ) as client:
                # Start the deep research interaction
                async def _start_interaction():
                    r = await client.post(
                        f"{GEMINI_BASE_URL}/interactions",
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
                    r = await client.get(f"{GEMINI_BASE_URL}/{interaction_name}")
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
                    status = result.get("status")

                    if status in _FAILED_STATUSES:
                        return OwlResponse(
                            model_name=self.model_name,
                            source="google-deep",
                            text="",
                            error=f"deep research {status}: {_failure_reason(result)}",
                        )

                    if status == "completed":
                        elapsed = time.monotonic() - t0
                        return OwlResponse(
                            model_name=self.model_name,
                            source="google-deep",
                            text=extract_text(result),
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
                error=describe_error(e),
            )
