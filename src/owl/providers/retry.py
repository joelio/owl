"""Retry logic for rate-limited API calls."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RETRIES = 2
RETRY_DELAYS = [2.0, 5.0]


async def with_retry(
    fn: Callable[[], Coroutine[None, None, T]],
    retryable_status_codes: tuple[int, ...] = (429, 502, 503),
) -> T:
    """Retry an async function on transient failures."""
    import httpx

    for attempt in range(MAX_RETRIES + 1):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in retryable_status_codes and attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                logger.info(
                    "Retrying %s after %ss (status %s)",
                    e.request.url,
                    delay,
                    e.response.status_code,
                )
                await asyncio.sleep(delay)
            else:
                raise
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                logger.info("Retrying after timeout (%ss delay)", delay)
                await asyncio.sleep(delay)
            else:
                raise
    raise RuntimeError("Unreachable")
