"""OpenAI Deep Research provider (o3-deep-research, o4-mini-deep-research)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .http_base import HttpProvider
from .retry import with_retry

OPENAI_BASE_URL = "https://api.openai.com/v1"

# Model name mapping
MODEL_MAP = {
    "o3-deep-research": "o3-deep-research",
    "o4-mini-deep-research": "o4-mini-deep-research",
}

# Deep research models accept only web_search_preview, code_interpreter, mcp
# and file_search.  The plain "web_search" tool is rejected with a 400.
# https://developers.openai.com/api/docs/guides/deep-research
WEB_SEARCH_TOOL = "web_search_preview"

POLL_INTERVAL = 5  # seconds between polls
MAX_RESEARCH_SECONDS = 1800  # deep research can run for tens of minutes

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "incomplete"})


class OpenAIDeepProvider(HttpProvider):
    source = "openai-deep"
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_model = MODEL_MAP.get(model_name, model_name)

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.api_model,
            "input": prompt,
            "tools": [{"type": WEB_SEARCH_TOOL}],
            # Deep research outruns any sane request timeout, so start the
            # job and poll rather than holding a socket open for it.
            "background": True,
        }
        if system_prompt:
            payload["instructions"] = system_prompt
        return {
            "url": f"{OPENAI_BASE_URL}/responses",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "json": payload,
        }

    async def follow_up(
        self, client: httpx.AsyncClient, data: dict[str, Any], api_key: str
    ) -> dict[str, Any]:
        """Poll a backgrounded response until it reaches a terminal status."""
        if data.get("status") in _TERMINAL_STATUSES:
            return data

        response_id = data.get("id")
        if not response_id:
            raise ValueError(f"unexpected response (no id): {str(data)[:200]}")

        deadline = time.monotonic() + MAX_RESEARCH_SECONDS

        async def _poll() -> httpx.Response:
            r = await client.get(
                f"{OPENAI_BASE_URL}/responses/{response_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            r.raise_for_status()
            return r

        while time.monotonic() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            # Retry transient poll failures so a blip does not discard
            # research that may already be many minutes in.
            data = (await with_retry(_poll)).json()
            if data.get("status") in _TERMINAL_STATUSES:
                return data

        raise TimeoutError(f"deep research did not finish within {MAX_RESEARCH_SECONDS}s")

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        status = data.get("status")
        if status and status != "completed":
            raise ValueError(f"deep research {status}: {_failure_reason(data)}")

        text_parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", ""))
        return {"text": "\n".join(text_parts)}


def _failure_reason(data: dict[str, Any]) -> str:
    """Best-effort reason for a deep research response that did not complete."""
    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    incomplete = data.get("incomplete_details")
    if isinstance(incomplete, dict) and incomplete.get("reason"):
        return str(incomplete["reason"])
    return "no reason given"
