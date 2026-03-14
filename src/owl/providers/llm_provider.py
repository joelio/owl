"""Provider that wraps Simon Willison's llm library."""

from __future__ import annotations

import asyncio
import time

from .base import OwlResponse, Provider


class LlmProvider(Provider):
    def __init__(self, model_id: str):
        self.model_id = model_id

    async def query(self, prompt: str, system_prompt: str | None = None) -> OwlResponse:
        try:
            import llm

            model = llm.get_model(self.model_id)

            def _run() -> str:
                return model.prompt(prompt, system=system_prompt).text()

            t0 = time.monotonic()
            response = await asyncio.to_thread(_run)
            elapsed = time.monotonic() - t0

            return OwlResponse(
                model_name=self.model_id,
                source="llm",
                text=response,
                elapsed_seconds=round(elapsed, 1),
            )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_id,
                source="llm",
                text="",
                error=str(e),
            )
