"""Base provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
    """A model owl can put a prompt to.

    ``model_name`` is declared here because the registry builds every
    provider as ``cls(model_name=...)``; leaving it to the subclasses made
    that call unverifiable.
    """

    model_name: str

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    async def query(self, prompt: str, system_prompt: str | None = None) -> OwlResponse:
        """Send a prompt and return the response."""
        ...
