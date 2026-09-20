"""Tests for providers."""

from __future__ import annotations

import pytest

from owl.config import CouncilMember
from owl.providers.base import OwlResponse
from owl.providers.deepseek import DeepSeekProvider
from owl.providers.google_deep import GoogleDeepProvider
from owl.providers.http_base import parse_chat_message
from owl.providers.llm_provider import LlmProvider
from owl.providers.openrouter import OpenRouterProvider
from owl.providers.registry import get_provider
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
            model_name="perplexity/sonar-reasoning-pro",
            source="openrouter",
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

    def test_openrouter_provider(self):
        member = CouncilMember(name="anthropic/claude-sonnet-4.6", source="openrouter")
        provider = get_provider(member)
        assert isinstance(provider, OpenRouterProvider)
        assert provider.model_name == "anthropic/claude-sonnet-4.6"

    def test_openrouter_fusion_provider(self):
        member = CouncilMember(name="openrouter/fusion", source="openrouter")
        provider = get_provider(member)
        assert isinstance(provider, OpenRouterProvider)
        assert provider.model_name == "openrouter/fusion"

    def test_dead_sources_are_gone(self):
        """The retired deep-research adapters must not resolve any more."""
        for source in ("openai-deep", "perplexity"):
            with pytest.raises(ValueError, match="Unknown source"):
                get_provider(CouncilMember(name="whatever", source=source))

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
        r = OwlResponse(model_name="gpt-5", source="llm", text="Hello", elapsed_seconds=1.5)
        assert r.elapsed_seconds == 1.5

    def test_response_elapsed_defaults_none(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="Hello")
        assert r.elapsed_seconds is None


class TestSystemPromptWiring:
    """Verify providers accept system_prompt without error."""

    @pytest.fixture(autouse=True)
    def clear_keys(self, monkeypatch):
        for key in [
            "OPENROUTER_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "XAI_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

    @pytest.mark.asyncio
    async def test_openrouter_accepts_system_prompt(self):
        provider = OpenRouterProvider("openrouter/fusion")
        # Will fail on missing key, but should not fail on the system_prompt arg
        response = await provider.query("test", system_prompt="Be helpful")
        assert response.error == "OPENROUTER_API_KEY not set"

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
            "OPENROUTER_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "XAI_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

    @pytest.mark.asyncio
    async def test_openrouter_no_key(self):
        provider = OpenRouterProvider("openrouter/fusion")
        response = await provider.query("test")
        assert response.error == "OPENROUTER_API_KEY not set"

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


class TestXAIModelName:
    """The xAI provider must send the configured model, not a hardcoded one."""

    @pytest.fixture(autouse=True)
    def set_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")

    def _reply(self, text="hi"):
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    @pytest.mark.asyncio
    async def test_default_model_maps_to_api_id(self, httpx_mock):
        httpx_mock.add_response(json=self._reply())
        provider = XAIProvider()  # grok-agentic
        resp = await provider.query("q")
        assert resp.text == "hi"
        import json

        body = json.loads(httpx_mock.get_request().content)
        assert body["model"] == "grok-4.5"

    @pytest.mark.asyncio
    async def test_custom_model_is_passed_through(self, httpx_mock):
        httpx_mock.add_response(json=self._reply())
        provider = XAIProvider("grok-4.3")
        await provider.query("q")
        import json

        body = json.loads(httpx_mock.get_request().content)
        assert body["model"] == "grok-4.3"


class TestProviderRetry:
    """Providers must retry transient failures via with_retry."""

    @pytest.fixture(autouse=True)
    def fast_retry(self, monkeypatch):
        # Avoid real 2s/5s sleeps during the test.
        monkeypatch.setattr("owl.providers.retry.RETRY_DELAYS", [0.0, 0.0])

    @pytest.mark.asyncio
    async def test_openrouter_retries_on_429(self, httpx_mock, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(json={"choices": [{"message": {"content": "ok"}}]})
        provider = OpenRouterProvider("openrouter/fusion")
        resp = await provider.query("q")
        assert resp.error is None
        assert resp.text == "ok"
        assert len(httpx_mock.get_requests()) == 2


class TestSafeParsing:
    """Malformed API responses must yield a clean error, not a crash."""

    @pytest.fixture(autouse=True)
    def fast_retry(self, monkeypatch):
        monkeypatch.setattr("owl.providers.retry.RETRY_DELAYS", [0.0, 0.0])

    @pytest.mark.asyncio
    async def test_openrouter_malformed_response_is_error(self, httpx_mock, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        httpx_mock.add_response(json={"unexpected": "shape"})
        resp = await OpenRouterProvider("openrouter/fusion").query("q")
        assert resp.text == ""
        assert resp.error is not None
        assert "unexpected response" in resp.error

    @pytest.mark.asyncio
    async def test_deepseek_extracts_reasoning(self, httpx_mock, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        httpx_mock.add_response(
            json={"choices": [{"message": {"content": "answer", "reasoning_content": "because"}}]}
        )
        resp = await DeepSeekProvider().query("q")
        assert resp.text == "answer"
        assert resp.reasoning == "because"

    @pytest.mark.asyncio
    async def test_openrouter_extracts_citations_from_annotations(self, httpx_mock, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        httpx_mock.add_response(
            json={
                "choices": [
                    {
                        "message": {
                            "content": "answer",
                            "annotations": [
                                {"url_citation": {"url": "https://a.example"}},
                                {"url_citation": {"url": "https://b.example"}},
                                {"url_citation": {"url": "https://a.example"}},
                            ],
                        }
                    }
                ],
            }
        )
        resp = await OpenRouterProvider("openrouter/fusion").query("q")
        assert resp.citations == ["https://a.example", "https://b.example"]

    @pytest.mark.asyncio
    async def test_openrouter_no_annotations_means_no_citations(self, httpx_mock, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        httpx_mock.add_response(json={"choices": [{"message": {"content": "answer"}}]})
        resp = await OpenRouterProvider("openai/gpt-5").query("q")
        assert resp.citations is None


class TestGoogleDeepPolling:
    """Gemini provider starts an interaction then polls to completion."""

    @pytest.fixture(autouse=True)
    def fast_poll(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setattr("owl.providers.google_deep.POLL_INTERVAL", 0)
        monkeypatch.setattr("owl.providers.retry.RETRY_DELAYS", [0.0, 0.0])

    @pytest.mark.asyncio
    async def test_completes_after_polling(self, httpx_mock):
        httpx_mock.add_response(json={"name": "interactions/abc"})  # start
        httpx_mock.add_response(
            json={
                "status": "completed",
                "steps": [
                    {"type": "model_output", "content": [{"type": "text", "text": "the report"}]}
                ],
            }
        )  # poll -> completed
        resp = await GoogleDeepProvider().query("q")
        assert resp.error is None
        assert resp.text == "the report"

    @pytest.mark.asyncio
    async def test_missing_interaction_name_is_error(self, httpx_mock):
        httpx_mock.add_response(json={})  # start returns no name
        resp = await GoogleDeepProvider().query("q")
        assert resp.text == ""
        assert resp.error == "No interaction name returned"


class TestParseChatMessage:
    """parse_chat_message is a pure utility — unit-test it directly."""

    def test_extracts_message(self):
        msg = parse_chat_message({"choices": [{"message": {"content": "hi"}}]})
        assert msg["content"] == "hi"

    def test_data_not_dict(self):
        with pytest.raises(ValueError, match="not an object"):
            parse_chat_message(["not", "a", "dict"])

    def test_choices_not_a_list(self):
        with pytest.raises(ValueError, match="no choices"):
            parse_chat_message({"choices": "not-a-list"})

    def test_empty_choices(self):
        with pytest.raises(ValueError, match="no choices"):
            parse_chat_message({"choices": []})

    def test_choice_not_a_dict(self):
        with pytest.raises(ValueError, match="choice is not a dict"):
            parse_chat_message({"choices": ["oops"]})

    def test_message_not_a_dict(self):
        with pytest.raises(ValueError, match="no message"):
            parse_chat_message({"choices": [{"message": "oops"}]})

    def test_missing_content(self):
        with pytest.raises(ValueError, match="no content"):
            parse_chat_message({"choices": [{"message": {}}]})


class TestGoogleDeepAuth:
    """The Gemini key must travel in a header, never in the URL.

    A key in the query string ends up inside str(HTTPStatusError), which owl
    surfaces as OwlResponse.error and posts to GitHub issues.
    """

    @pytest.fixture(autouse=True)
    def fast_poll(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "secret-key-value")
        monkeypatch.setattr("owl.providers.google_deep.POLL_INTERVAL", 0)
        monkeypatch.setattr("owl.providers.retry.RETRY_DELAYS", [0.0, 0.0])

    @pytest.mark.asyncio
    async def test_key_sent_as_header_not_query_param(self, httpx_mock):
        httpx_mock.add_response(json={"name": "interactions/abc"})
        httpx_mock.add_response(
            json={
                "status": "completed",
                "steps": [
                    {"type": "model_output", "content": [{"type": "text", "text": "report"}]}
                ],
            }
        )

        await GoogleDeepProvider().query("q")

        requests = httpx_mock.get_requests()
        assert requests, "expected the provider to make requests"
        for request in requests:
            assert request.headers.get("x-goog-api-key") == "secret-key-value"
            assert "secret-key-value" not in str(request.url)

    @pytest.mark.asyncio
    async def test_http_error_does_not_leak_key(self, httpx_mock):
        httpx_mock.add_response(status_code=401)

        resp = await GoogleDeepProvider().query("q")

        assert resp.error is not None
        assert "secret-key-value" not in resp.error
        assert "401" in resp.error


class TestDeepSeekContract:
    """The legacy deepseek-chat / deepseek-reasoner names were discontinued
    on 2026-07-24; V4 needs thinking enabled explicitly."""

    def test_default_model_is_current(self):
        request = DeepSeekProvider().build_request("q", None, "k")
        assert request["json"]["model"] == "deepseek-v4-flash"

    def test_legacy_reasoner_name_is_remapped(self):
        request = DeepSeekProvider("deepseek-reasoner").build_request("q", None, "k")
        assert request["json"]["model"] == "deepseek-v4-flash"

    def test_thinking_is_enabled(self):
        request = DeepSeekProvider().build_request("q", None, "k")
        assert request["json"]["thinking"] == {"type": "enabled"}

    def test_unknown_model_passes_through(self):
        request = DeepSeekProvider("deepseek-v4-pro").build_request("q", None, "k")
        assert request["json"]["model"] == "deepseek-v4-pro"


class TestXAIContract:
    """Chat Completions is a deprecated endpoint supporting function calling
    only; server-side web/X search is native to the Responses API."""

    def test_uses_the_responses_endpoint(self):
        request = XAIProvider().build_request("q", None, "k")
        assert request["url"] == "https://api.x.ai/v1/responses"

    def test_friendly_name_maps_to_a_real_model(self):
        request = XAIProvider("grok-agentic").build_request("q", None, "k")
        assert request["json"]["model"] == "grok-4.5"

    def test_sends_responses_style_input_not_messages(self):
        payload = XAIProvider().build_request("q", None, "k")["json"]
        assert payload["input"] == [{"role": "user", "content": "q"}]
        assert "messages" not in payload

    def test_no_chain_limit_parameter(self):
        # chain_limit is not an xAI parameter and never was.
        assert "chain_limit" not in XAIProvider().build_request("q", None, "k")["json"]

    def test_system_prompt_becomes_instructions(self):
        payload = XAIProvider().build_request("q", "be terse", "k")["json"]
        assert payload["instructions"] == "be terse"

    def test_parses_responses_output(self):
        parsed = XAIProvider().parse_response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "the answer"}],
                    }
                ]
            }
        )
        assert parsed["text"] == "the answer"


class TestGoogleDeepResponseShape:
    """Interactions report a `status` string, not a `done` boolean, and the
    text lives in steps[].content[].text."""

    @pytest.fixture(autouse=True)
    def fast_poll(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setattr("owl.providers.google_deep.POLL_INTERVAL", 0)
        monkeypatch.setattr("owl.providers.retry.RETRY_DELAYS", [0.0, 0.0])

    def test_extract_text_reads_steps(self):
        from owl.providers.google_deep import extract_text

        result = {
            "status": "completed",
            "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "part one"}]},
                {"type": "model_output", "content": [{"type": "text", "text": "part two"}]},
            ],
        }
        assert extract_text(result) == "part one\npart two"

    def test_extract_text_ignores_non_text_content(self):
        from owl.providers.google_deep import extract_text

        result = {"steps": [{"content": [{"type": "reasoning"}, {"type": "text", "text": "keep"}]}]}
        assert extract_text(result) == "keep"

    @pytest.mark.asyncio
    async def test_completed_status_returns_the_report(self, httpx_mock):
        httpx_mock.add_response(json={"name": "interactions/abc"})
        httpx_mock.add_response(json={"status": "in_progress"})
        httpx_mock.add_response(
            json={
                "status": "completed",
                "steps": [
                    {"type": "model_output", "content": [{"type": "text", "text": "the report"}]}
                ],
            }
        )

        resp = await GoogleDeepProvider().query("q")

        assert resp.error is None
        assert resp.text == "the report"

    @pytest.mark.asyncio
    async def test_failed_status_is_an_error(self, httpx_mock):
        httpx_mock.add_response(json={"name": "interactions/abc"})
        httpx_mock.add_response(json={"status": "failed", "error": {"message": "quota exhausted"}})

        resp = await GoogleDeepProvider().query("q")

        assert resp.text == ""
        assert "failed" in (resp.error or "")
        assert "quota exhausted" in (resp.error or "")


class TestProviderConstruction:
    """Every provider is built as cls(model_name=...) by the registry."""

    @pytest.mark.parametrize(
        "cls",
        [LlmProvider, OpenRouterProvider, DeepSeekProvider, XAIProvider],
    )
    def test_accepts_the_model_name_keyword(self, cls):
        assert cls(model_name="some-model").model_name == "some-model"

    def test_llm_provider_keeps_model_id_alias(self):
        provider = LlmProvider(model_name="gpt-5")
        assert provider.model_id == "gpt-5"


class TestOpenRouterAttribution:
    """A council member declared as a plain OpenRouter slug must be sent
    unchanged and attributed back to that exact slug."""

    @pytest.fixture(autouse=True)
    def set_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "slug",
        ["openrouter/fusion", "anthropic/claude-sonnet-4.6", "openai/gpt-5"],
    )
    async def test_slug_sent_and_attributed(self, httpx_mock, slug):
        import json

        httpx_mock.add_response(json={"choices": [{"message": {"content": "hi"}}]})
        resp = await OpenRouterProvider(slug).query("q")

        request = httpx_mock.get_request()
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert json.loads(request.content)["model"] == slug
        # Attribution: the response carries the exact slug that was asked for.
        assert resp.model_name == slug
        assert resp.source == "openrouter"

    @pytest.mark.asyncio
    async def test_fusion_reply_is_not_relabelled_by_upstream(self, httpx_mock):
        """Fusion blends several models; owl still attributes it to the declared
        slug rather than whatever upstream name comes back."""
        httpx_mock.add_response(
            json={
                "model": "some/underlying-model",
                "choices": [{"message": {"content": "blended answer"}}],
            }
        )
        resp = await OpenRouterProvider("openrouter/fusion").query("q")
        assert resp.model_name == "openrouter/fusion"
