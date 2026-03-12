"""Tests for providers."""

from __future__ import annotations

import pytest

from owl.config import CouncilMember
from owl.providers.base import OwlResponse
from owl.providers.registry import get_provider
from owl.providers.llm_provider import LlmProvider
from owl.providers.openai_deep import OpenAIDeepProvider
from owl.providers.perplexity import PerplexityProvider
from owl.providers.deepseek import DeepSeekProvider
from owl.providers.google_deep import GoogleDeepProvider
from owl.providers.xai import XAIProvider


class TestOwlResponse:
    def test_successful_response(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="Hello world")
        assert r.text == "Hello world"
        assert r.error is None
        assert r.citations is None

    def test_error_response(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="", error="API error")
        assert r.error == "API error"

    def test_response_with_citations(self):
        r = OwlResponse(
            model_name="sonar",
            source="perplexity",
            text="Answer",
            citations=["https://example.com"],
        )
        assert r.citations == ["https://example.com"]

    def test_response_with_reasoning(self):
        r = OwlResponse(
            model_name="deepseek-reasoner",
            source="deepseek",
            text="Answer",
            reasoning="Step 1: ...",
        )
        assert r.reasoning == "Step 1: ..."


class TestRegistry:
    def test_llm_provider(self):
        member = CouncilMember(name="gpt-5", source="llm")
        provider = get_provider(member)
        assert isinstance(provider, LlmProvider)
        assert provider.model_id == "gpt-5"

    def test_openai_deep_provider(self):
        member = CouncilMember(name="o3-deep-research", source="openai-deep")
        provider = get_provider(member)
        assert isinstance(provider, OpenAIDeepProvider)

    def test_perplexity_provider(self):
        member = CouncilMember(name="sonar-deep-research", source="perplexity")
        provider = get_provider(member)
        assert isinstance(provider, PerplexityProvider)

    def test_deepseek_provider(self):
        member = CouncilMember(name="deepseek-reasoner", source="deepseek")
        provider = get_provider(member)
        assert isinstance(provider, DeepSeekProvider)

    def test_google_deep_provider(self):
        member = CouncilMember(name="gemini-deep-research", source="google-deep")
        provider = get_provider(member)
        assert isinstance(provider, GoogleDeepProvider)

    def test_xai_provider(self):
        member = CouncilMember(name="grok-agentic", source="xai")
        provider = get_provider(member)
        assert isinstance(provider, XAIProvider)

    def test_unknown_source_raises(self):
        member = CouncilMember(name="unknown", source="nope")
        with pytest.raises(ValueError, match="Unknown source"):
            get_provider(member)


class TestElapsedSeconds:
    def test_response_stores_elapsed(self):
        r = OwlResponse(
            model_name="gpt-5", source="llm", text="Hello", elapsed_seconds=1.5
        )
        assert r.elapsed_seconds == 1.5

    def test_response_elapsed_defaults_none(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="Hello")
        assert r.elapsed_seconds is None


class TestSystemPromptWiring:
    """Verify providers accept system_prompt without error."""

    @pytest.fixture(autouse=True)
    def clear_keys(self, monkeypatch):
        for key in [
            "OPENAI_API_KEY",
            "PERPLEXITY_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "XAI_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

    @pytest.mark.asyncio
    async def test_openai_deep_accepts_system_prompt(self):
        provider = OpenAIDeepProvider("o3-deep-research")
        # Will fail on missing key, but should not fail on the system_prompt arg
        response = await provider.query("test", system_prompt="Be helpful")
        assert response.error == "OPENAI_API_KEY not set"

    @pytest.mark.asyncio
    async def test_perplexity_accepts_system_prompt(self):
        provider = PerplexityProvider()
        response = await provider.query("test", system_prompt="Be helpful")
        assert response.error == "PERPLEXITY_API_KEY not set"

    @pytest.mark.asyncio
    async def test_deepseek_accepts_system_prompt(self):
        provider = DeepSeekProvider()
        response = await provider.query("test", system_prompt="Be helpful")
        assert response.error == "DEEPSEEK_API_KEY not set"

    @pytest.mark.asyncio
    async def test_google_deep_accepts_system_prompt(self):
        provider = GoogleDeepProvider()
        response = await provider.query("test", system_prompt="Be helpful")
        assert response.error == "GOOGLE_API_KEY not set"

    @pytest.mark.asyncio
    async def test_xai_accepts_system_prompt(self):
        provider = XAIProvider()
        response = await provider.query("test", system_prompt="Be helpful")
        assert response.error == "XAI_API_KEY not set"


class TestProvidersMissingKeys:
    """All deep research providers should return error responses when keys are missing."""

    @pytest.fixture(autouse=True)
    def clear_keys(self, monkeypatch):
        for key in [
            "OPENAI_API_KEY",
            "PERPLEXITY_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "XAI_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

    @pytest.mark.asyncio
    async def test_openai_deep_no_key(self):
        provider = OpenAIDeepProvider("o3-deep-research")
        response = await provider.query("test")
        assert response.error == "OPENAI_API_KEY not set"

    @pytest.mark.asyncio
    async def test_perplexity_no_key(self):
        provider = PerplexityProvider()
        response = await provider.query("test")
        assert response.error == "PERPLEXITY_API_KEY not set"

    @pytest.mark.asyncio
    async def test_google_deep_no_key(self):
        provider = GoogleDeepProvider()
        response = await provider.query("test")
        assert response.error == "GOOGLE_API_KEY not set"

    @pytest.mark.asyncio
    async def test_deepseek_no_key(self):
        provider = DeepSeekProvider()
        response = await provider.query("test")
        assert response.error == "DEEPSEEK_API_KEY not set"

    @pytest.mark.asyncio
    async def test_xai_no_key(self):
        provider = XAIProvider()
        response = await provider.query("test")
        assert response.error == "XAI_API_KEY not set"
