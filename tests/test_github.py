"""Tests for GitHub integration."""

from __future__ import annotations

import pytest

from owl.github import _build_consolidated_comment, _format_response_section
from owl.providers.base import OwlResponse


class TestFormatResponseSection:
    def test_basic_response(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="Hello world")
        section = _format_response_section(r)
        assert "gpt-5" in section
        assert "Hello world" in section
        assert "<details>" in section
        assert "</details>" in section

    def test_response_with_citations(self):
        r = OwlResponse(
            model_name="sonar",
            source="perplexity",
            text="Answer",
            citations=["https://example.com", "https://other.com"],
        )
        section = _format_response_section(r)
        assert "Sources" in section
        assert "https://example.com" in section
        assert "https://other.com" in section

    def test_response_with_reasoning(self):
        r = OwlResponse(
            model_name="deepseek-reasoner",
            source="deepseek",
            text="Final answer",
            reasoning="Step 1: think\nStep 2: conclude",
        )
        section = _format_response_section(r)
        assert section.count("<details>") == 2  # outer + reasoning
        assert "Reasoning" in section
        assert "Step 1: think" in section
        assert "Final answer" in section

    def test_source_label_for_llm(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="test")
        section = _format_response_section(r)
        assert "llm plugin" in section

    def test_source_label_for_deep_research(self):
        r = OwlResponse(model_name="o3", source="openai-deep", text="test")
        section = _format_response_section(r)
        assert "openai-deep" in section

    def test_timing_shown(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="test", elapsed_seconds=2.3)
        section = _format_response_section(r)
        assert "2.3s" in section


class TestBuildConsolidatedComment:
    def test_single_response(self):
        responses = [OwlResponse(model_name="gpt-5", source="llm", text="Hello")]
        bodies = _build_consolidated_comment(responses)
        assert len(bodies) == 1
        assert "Council Response" in bodies[0]
        assert "1 of 1 members responded" in bodies[0]
        assert "gpt-5" in bodies[0]
        assert "Parliament of Owls" in bodies[0]

    def test_error_responses_shown(self):
        responses = [
            OwlResponse(model_name="gpt-5", source="llm", text="OK"),
            OwlResponse(model_name="bad-model", source="llm", text="", error="API timeout"),
        ]
        bodies = _build_consolidated_comment(responses)
        combined = "\n".join(bodies)
        assert "1 of 2 members responded" in combined
        assert "Errors" in combined
        assert "API timeout" in combined

    def test_all_collapsed(self):
        """All successful responses should be inside <details> tags."""
        responses = [
            OwlResponse(model_name="model-a", source="llm", text="Answer A"),
            OwlResponse(model_name="model-b", source="llm", text="Answer B"),
        ]
        bodies = _build_consolidated_comment(responses)
        combined = "\n".join(bodies)
        assert combined.count("<details>") >= 2
        assert combined.count("</details>") >= 2

    def test_overflow_splits_into_multiple_comments(self):
        """When responses exceed the char limit, they split into multiple comments."""
        # Create responses that are large enough to exceed the limit
        big_text = "x" * 20_000
        responses = [
            OwlResponse(model_name=f"model-{i}", source="llm", text=big_text) for i in range(5)
        ]
        bodies = _build_consolidated_comment(responses)
        assert len(bodies) > 1
        for body in bodies:
            assert len(body) <= 62_000 + 5_000  # allow some slack for headers/footers
            assert "Parliament of Owls" in body  # each chunk has footer

    def test_single_comment_when_small(self):
        """Small responses fit in one comment."""
        responses = [
            OwlResponse(model_name="model-a", source="llm", text="Short answer"),
        ]
        bodies = _build_consolidated_comment(responses)
        assert len(bodies) == 1


class TestGitHubTimeouts:
    """GitHub HTTP calls must set a timeout so they cannot hang forever."""

    @pytest.mark.asyncio
    async def test_create_issue_sets_timeout(self, httpx_mock, monkeypatch):
        import owl.github as gh

        captured = {}
        real_client = gh.httpx.AsyncClient

        def spy(*args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return real_client(*args, **kwargs)

        monkeypatch.setattr(gh.httpx, "AsyncClient", spy)
        monkeypatch.setattr(gh, "_get_token", lambda: "tok")
        httpx_mock.add_response(json={"number": 7})

        num = await gh.create_issue("o/r", "title", "body")
        assert num == 7
        assert captured["timeout"] == 30.0


class TestOversizedResponses:
    """No comment body may exceed GitHub's hard limit.

    GitHub rejects an oversized body with a 422, and because every response
    is posted in the same pass, one huge response used to lose them all.
    """

    GITHUB_HARD_LIMIT = 65_536

    def test_single_huge_response_is_truncated(self):
        huge = OwlResponse(model_name="big", source="llm", text="x" * 200_000)
        bodies = _build_consolidated_comment([huge])
        assert len(bodies) == 1
        assert len(bodies[0]) <= self.GITHUB_HARD_LIMIT
        assert "truncated" in bodies[0]

    def test_truncated_section_keeps_details_balanced(self):
        huge = OwlResponse(model_name="big", source="llm", text="x" * 200_000)
        body = _build_consolidated_comment([huge])[0]
        assert body.count("<details>") == body.count("</details>")

    def test_reasoning_is_dropped_before_the_answer_is_cut(self):
        response = OwlResponse(
            model_name="reasoner",
            source="deepseek",
            text="the short answer",
            reasoning="y" * 200_000,
        )
        body = _build_consolidated_comment([response])[0]
        assert len(body) <= self.GITHUB_HARD_LIMIT
        assert "the short answer" in body
        assert "yyyy" not in body

    def test_several_huge_responses_all_fit(self):
        responses = [
            OwlResponse(model_name=f"m{i}", source="llm", text="z" * 100_000) for i in range(4)
        ]
        bodies = _build_consolidated_comment(responses)
        assert len(bodies) == 4
        for body in bodies:
            assert len(body) <= self.GITHUB_HARD_LIMIT

    def test_normal_responses_are_untouched(self):
        responses = [OwlResponse(model_name="m", source="llm", text="a normal answer")]
        body = _build_consolidated_comment(responses)[0]
        assert "truncated" not in body
        assert "a normal answer" in body
