"""Provider that wraps Simon Willison's llm library."""

from __future__ import annotations

import asyncio

from .base import OwlResponse, Provider


class LlmProvider(Provider):
    def __init__(self, model_id: str):
        self.model_id = model_id

    async def query(self, prompt: str) -> OwlResponse:
        try:
            import llm

            model = llm.get_model(self.model_id)
            # llm's async support - run in executor if no native async
            response = await asyncio.to_thread(lambda: model.prompt(prompt).text())
            return OwlResponse(
                model_name=self.model_id,
                source="llm",
                text=response,
            )
        except Exception as e:
            return OwlResponse(
                model_name=self.model_id,
                source="llm",
                text="",
                error=str(e),
            )
