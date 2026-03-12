"""Base provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OwlResponse:
    model_name: str
    source: str
    text: str
    error: str | None = None
    citations: list[str] | None = None
    reasoning: str | None = None
    elapsed_seconds: float | None = None


class Provider(ABC):
    @abstractmethod
    async def query(self, prompt: str, system_prompt: str | None = None) -> OwlResponse:
        """Send a prompt and return the response."""
        ...
