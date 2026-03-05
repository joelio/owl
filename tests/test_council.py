"""Tests for the council engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from owl.config import Config, CouncilMember
from owl.council import convene, query_member
from owl.providers.base import OwlResponse


@pytest.mark.asyncio
async def test_convene_empty_council():
    config = Config(council=[])
    results = await convene("test prompt", config)
    assert results == []


@pytest.mark.asyncio
async def test_convene_parallel_dispatch():
    """Verify all members are queried in parallel."""
    config = Config(
        council=[
            CouncilMember("model-a", "llm"),
            CouncilMember("model-b", "llm"),
        ]
    )

    mock_response_a = OwlResponse(model_name="model-a", source="llm", text="Response A")
    mock_response_b = OwlResponse(model_name="model-b", source="llm", text="Response B")

    with patch("owl.council.query_member", new_callable=AsyncMock) as mock_qm:
        mock_qm.side_effect = [mock_response_a, mock_response_b]
        results = await convene("test", config)

    assert len(results) == 2
    assert mock_qm.call_count == 2


@pytest.mark.asyncio
async def test_query_member_uses_correct_provider():
    member = CouncilMember("o3-deep-research", "openai-deep")

    with patch("owl.council.get_provider") as mock_get:
        mock_provider = AsyncMock()
        mock_provider.query.return_value = OwlResponse(
            model_name="o3-deep-research", source="openai-deep", text="result"
        )
        mock_get.return_value = mock_provider

        result = await query_member(member, "test prompt")

    mock_get.assert_called_once_with(member)
    mock_provider.query.assert_called_once_with("test prompt")
    assert result.text == "result"
