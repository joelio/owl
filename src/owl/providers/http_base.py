"""Shared base for HTTP-based providers.

Most providers follow the same shape: check an API key, POST a JSON payload
once, then parse the JSON response.  This base class captures that flow
(including retry, timing, logging and error handling) so each provider only
has to declare *what* request to send and *how* to read the reply.

Providers with a fundamentally different protocol (e.g. Gemini's async polling
in ``google_deep``) do not use this base.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from .base import OwlResponse, Provider
from .errors import describe_error
from .retry import with_retry

logger = logging.getLogger(__name__)


def build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    """Build a chat-completions ``messages`` array."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def parse_chat_message(data: dict[str, Any]) -> dict[str, Any]:
    """Safely pull the first ``choices[].message`` from a chat-completions reply.

    Raises ``ValueError`` with a truncated snippet of the payload if the shape
    is unexpected, so the caller surfaces a clear error instead of a raw
    ``KeyError``/``IndexError``.
    """
    if not isinstance(data, dict):
        raise ValueError(f"unexpected response (not an object): {str(data)[:200]}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"unexpected response (no choices): {str(data)[:200]}")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError(f"unexpected response (choice is not a dict): {str(data)[:200]}")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError(f"unexpected response (no message): {str(data)[:200]}")
    if message.get("content") is None:
        raise ValueError(f"unexpected response (no content): {str(data)[:200]}")
    return message


class HttpProvider(Provider):
    """Base for providers that make a single POST and parse the JSON reply.

    Subclasses set ``source`` and ``api_key_env`` class attributes and
    implement :meth:`build_request` and :meth:`parse_response`.
    """

    source: str = ""
    api_key_env: str = ""
    timeout: float = 300.0

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        """Return kwargs passed straight to ``client.post`` (url, headers, json...)."""
        raise NotImplementedError

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return a dict with ``text`` and optional ``citations``/``reasoning``."""
        raise NotImplementedError

    async def follow_up(
        self, client: httpx.AsyncClient, data: dict[str, Any], api_key: str
    ) -> dict[str, Any]:
        """Hook for APIs whose first reply is a job handle rather than an answer.

        Runs on the still-open client, before :meth:`parse_response`.  The
        default is a no-op, which is the single-POST case.
        """
        return data

    async def query(self, prompt: str, system_prompt: str | None = None) -> OwlResponse:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            return OwlResponse(
                model_name=self.model_name,
                source=self.source,
                text="",
                error=f"{self.api_key_env} not set",
            )

        try:
            t0 = time.monotonic()
            request = self.build_request(prompt, system_prompt, api_key)
            async with httpx.AsyncClient(timeout=self.timeout) as client:

                async def _do_request():
                    r = await client.post(**request)
                    r.raise_for_status()
                    return r

                resp = await with_retry(_do_request)
                data = await self.follow_up(client, resp.json(), api_key)

            parsed = self.parse_response(data)
            return OwlResponse(
                model_name=self.model_name,
                source=self.source,
                text=parsed.get("text", ""),
                citations=parsed.get("citations"),
                reasoning=parsed.get("reasoning"),
                elapsed_seconds=round(time.monotonic() - t0, 1),
            )
        except Exception as e:
            logger.exception("%s provider (%s) failed", self.source, self.model_name)
            return OwlResponse(
                model_name=self.model_name,
                source=self.source,
                text="",
                error=describe_error(e),
            )
