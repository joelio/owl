"""Core council engine - dispatches prompts to all members in parallel."""

from __future__ import annotations

import asyncio
import logging

from .config import Config, CouncilMember, load_config
from .providers.base import OwlResponse
from .providers.registry import get_provider

logger = logging.getLogger(__name__)

# Stagger requests to avoid hitting rate limits on free-tier models
STAGGER_DELAY = 0.3  # seconds between launching each request


async def query_member(member: CouncilMember, prompt: str) -> OwlResponse:
    """Query a single council member."""
    provider = get_provider(member)
    try:
        return await provider.query(prompt)
    except Exception as e:
        logger.exception("Unexpected error querying %s", member.name)
        return OwlResponse(
            model_name=member.name,
            source=member.source,
            text="",
            error=str(e),
        )


async def convene(prompt: str, config: Config | None = None) -> list[OwlResponse]:
    """Convene the council - query all members in parallel with staggered start."""
    if config is None:
        config = load_config()

    if not config.council:
        return []

    async def staggered_query(index: int, member: CouncilMember) -> OwlResponse:
        if index > 0:
            await asyncio.sleep(index * STAGGER_DELAY)
        return await query_member(member, prompt)

    tasks = [staggered_query(i, member) for i, member in enumerate(config.council)]
    return await asyncio.gather(*tasks)
