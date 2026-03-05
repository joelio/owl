"""Core council engine - dispatches prompts to all members in parallel."""

from __future__ import annotations

import asyncio

from .config import Config, CouncilMember, load_config
from .providers.base import OwlResponse
from .providers.registry import get_provider


async def query_member(member: CouncilMember, prompt: str) -> OwlResponse:
    """Query a single council member."""
    provider = get_provider(member)
    return await provider.query(prompt)


async def convene(prompt: str, config: Config | None = None) -> list[OwlResponse]:
    """Convene the council - query all members in parallel."""
    if config is None:
        config = load_config()

    if not config.council:
        return []

    tasks = [query_member(member, prompt) for member in config.council]
    return await asyncio.gather(*tasks)
