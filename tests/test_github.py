"""Tests for GitHub integration."""

from __future__ import annotations

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
        bodies = _build_consolidated_comment(responses, "test prompt")
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
        bodies = _build_consolidated_comment(responses, "test")
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
        bodies = _build_consolidated_comment(responses, "test")
        combined = "\n".join(bodies)
        assert combined.count("<details>") >= 2
        assert combined.count("</details>") >= 2
