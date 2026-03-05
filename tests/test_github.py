"""Tests for GitHub integration."""

from __future__ import annotations


from owl.github import _format_comment
from owl.providers.base import OwlResponse


class TestFormatComment:
    def test_basic_response(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="Hello world")
        comment = _format_comment(r)
        assert "## 🦉 gpt-5" in comment
        assert "Hello world" in comment
        assert "Parliament of Owls" in comment

    def test_error_response(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="", error="API timeout")
        comment = _format_comment(r)
        assert "**Error:** API timeout" in comment

    def test_response_with_citations(self):
        r = OwlResponse(
            model_name="sonar",
            source="perplexity",
            text="Answer",
            citations=["https://example.com", "https://other.com"],
        )
        comment = _format_comment(r)
        assert "Sources" in comment
        assert "https://example.com" in comment
        assert "https://other.com" in comment

    def test_response_with_reasoning(self):
        r = OwlResponse(
            model_name="deepseek-reasoner",
            source="deepseek",
            text="Final answer",
            reasoning="Step 1: think\nStep 2: conclude",
        )
        comment = _format_comment(r)
        assert "<details>" in comment
        assert "Reasoning" in comment
        assert "Step 1: think" in comment
        assert "Final answer" in comment

    def test_source_label_for_llm(self):
        r = OwlResponse(model_name="gpt-5", source="llm", text="test")
        comment = _format_comment(r)
        assert "llm plugin" in comment

    def test_source_label_for_deep_research(self):
        r = OwlResponse(model_name="o3", source="openai-deep", text="test")
        comment = _format_comment(r)
        assert "openai-deep" in comment
