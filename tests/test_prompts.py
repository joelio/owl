"""Tests for system prompt generation."""

from __future__ import annotations

import pytest

from owl.prompts import ResponseFormat, get_system_prompt


class TestResponseFormat:
    def test_enum_values(self):
        assert ResponseFormat.BRIEF.value == "brief"
        assert ResponseFormat.STANDARD.value == "standard"
        assert ResponseFormat.DETAILED.value == "detailed"


class TestGetSystemPrompt:
    @pytest.mark.parametrize("source", ["llm", "xai", "openrouter", "deepseek"])
    def test_standard_sources_contain_structure(self, source):
        prompt = get_system_prompt(source)
        assert "ANSWER:" in prompt
        assert "REASONING:" in prompt
        assert "CAVEATS:" in prompt
        assert "KEY POINTS:" in prompt

    def test_deep_research_sources_contain_synthesis(self):
        for source in ("google-deep",):
            prompt = get_system_prompt(source)
            assert "SYNTHESIS" in prompt
            assert "EVIDENCE" in prompt
            assert "COUNTERARGUMENTS" in prompt
            assert "SOURCES" in prompt

    def test_deep_research_ignores_format(self):
        """Deep research always returns the same template regardless of format."""
        brief = get_system_prompt("google-deep", ResponseFormat.BRIEF)
        detailed = get_system_prompt("google-deep", ResponseFormat.DETAILED)
        assert brief == detailed

    @pytest.mark.parametrize(
        "slug",
        ["openrouter/fusion", "perplexity/sonar-reasoning-pro", "openai/gpt-4o:online"],
    )
    def test_web_searching_slugs_get_citation_instructions(self, slug):
        prompt = get_system_prompt("openrouter", model_name=slug)
        assert "Cite at least 3" in prompt
        assert "[1]" in prompt

    def test_non_searching_slugs_are_not_asked_to_cite(self):
        """Asking a model without web search for sources just invites made-up URLs."""
        prompt = get_system_prompt("openrouter", model_name="openai/gpt-5")
        assert "Cite at least 3" not in prompt
        assert "ANSWER:" in prompt

    def test_retired_sources_fall_back_to_standard(self):
        """openai-deep / perplexity are gone; they must not get special handling."""
        for source in ("openai-deep", "perplexity"):
            assert get_system_prompt(source) == get_system_prompt("llm")

    def test_deepseek_skips_cot(self):
        prompt = get_system_prompt("deepseek")
        assert "chain-of-thought" in prompt.lower() or "Do NOT include chain" in prompt

    @pytest.mark.parametrize(
        "fmt,expected_fragment",
        [
            (ResponseFormat.BRIEF, "100-200 words"),
            (ResponseFormat.STANDARD, "250-400 words"),
            (ResponseFormat.DETAILED, "600-1000 words"),
        ],
    )
    def test_word_targets_per_format(self, fmt, expected_fragment):
        prompt = get_system_prompt("llm", fmt)
        assert expected_fragment in prompt

    def test_uncertainty_instruction_present(self):
        for source in ("llm", "openrouter", "deepseek", "xai"):
            prompt = get_system_prompt(source)
            assert "uncertain" in prompt.lower()

    def test_unknown_source_uses_standard(self):
        """Unknown sources fall back to the generic template."""
        prompt = get_system_prompt("some-future-provider")
        assert "ANSWER:" in prompt
