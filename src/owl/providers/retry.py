"""Retry logic for rate-limited and transiently failing API calls."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TypeVar

import httpx

from .errors import redact

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RETRIES = 2
RETRY_DELAYS = [2.0, 5.0]

# Jitter spreads retries out when a whole council is rate-limited at once and
# would otherwise march back in lockstep.
JITTER_RATIO = 0.25

# A server asking us to wait longer than this is not worth honouring inside a
# single query; give up and report instead of stalling the whole council.
MAX_RETRY_AFTER_SECONDS = 60.0

RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

# Connection resets and DNS blips are as transient as a 503, and providers
# under load produce plenty of both.
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


def _delay_for(attempt: int) -> float:
    """Backoff for ``attempt``, with jitter, safe beyond the delay table."""
    base = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
    return base + random.uniform(0, base * JITTER_RATIO)


def parse_retry_after(value: str | None) -> float | None:
    """Read a ``Retry-After`` header, which may be seconds or an HTTP date.

    Returns ``None`` when absent, unparseable, or beyond the ceiling we are
    willing to wait.
    """
    if not value:
        return None

    raw = value.strip()
    try:
        seconds = float(raw)
    except ValueError:
        try:
            target = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        now = datetime.now(timezone.utc) if target.tzinfo else datetime.now()
        seconds = (target - now).total_seconds()

    if seconds <= 0 or seconds > MAX_RETRY_AFTER_SECONDS:
        return None
    return seconds


async def with_retry(
    fn: Callable[[], Coroutine[None, None, T]],
    retryable_status_codes: tuple[int, ...] = RETRYABLE_STATUS_CODES,
) -> T:
    """Retry an async function on transient failures.

    Honours ``Retry-After`` when the server sends one, since a provider
    telling us exactly when to come back beats guessing.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in retryable_status_codes or attempt >= MAX_RETRIES:
                raise
            delay = parse_retry_after(e.response.headers.get("Retry-After")) or _delay_for(attempt)
            logger.info(
                "Retrying %s in %.1fs (status %s)",
                redact(str(e.request.url)),
                delay,
                e.response.status_code,
            )
            await asyncio.sleep(delay)
        except _RETRYABLE_EXCEPTIONS as e:
            if attempt >= MAX_RETRIES:
                raise
            delay = _delay_for(attempt)
            logger.info("Retrying in %.1fs after %s", delay, type(e).__name__)
            await asyncio.sleep(delay)

    raise RuntimeError("Unreachable")
