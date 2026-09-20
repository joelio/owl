"""Tests for model discovery."""

from __future__ import annotations

from owl.models import (
    DEEP_RESEARCH_KEY_MAP,
    DEEP_RESEARCH_MODELS,
    OPENROUTER_MODELS,
    discover_deep_research_models,
    discover_llm_models,
    discover_openrouter_models,
)


class TestDeepResearchModels:
    def test_all_deep_models_have_key_mapping(self):
        for model in DEEP_RESEARCH_MODELS:
            assert model.source in DEEP_RESEARCH_KEY_MAP, (
                f"Model {model.name} source '{model.source}' has no key mapping"
            )

    def test_discover_with_no_keys(self, monkeypatch):
        for env_var in DEEP_RESEARCH_KEY_MAP.values():
            monkeypatch.delenv(env_var, raising=False)
        result = discover_deep_research_models()
        assert result == []

    def test_discover_with_single_key(self, monkeypatch):
        for env_var in DEEP_RESEARCH_KEY_MAP.values():
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        result = discover_deep_research_models()
        names = [m.name for m in result]
        assert "deepseek-v4-flash" in names
        assert "deepseek-v4-pro" in names
        assert "gemini-deep-research" not in names

    def test_discover_with_all_keys(self, monkeypatch):
        for env_var in DEEP_RESEARCH_KEY_MAP.values():
            monkeypatch.setenv(env_var, "test-key")
        result = discover_deep_research_models()
        assert len(result) == len(DEEP_RESEARCH_MODELS)

    def test_all_deep_models_are_deep_category(self):
        for model in DEEP_RESEARCH_MODELS:
            assert model.category == "deep-research"

    def test_retired_adapters_are_gone(self):
        """The sunset OpenAI/Perplexity deep-research models must not be offered."""
        names = {m.name for m in DEEP_RESEARCH_MODELS}
        sources = {m.source for m in DEEP_RESEARCH_MODELS}
        assert names.isdisjoint(
            {"o3-deep-research", "o4-mini-deep-research", "sonar-deep-research"}
        )
        assert sources.isdisjoint({"openai-deep", "perplexity"})
        assert "openai-deep" not in DEEP_RESEARCH_KEY_MAP
        assert "perplexity" not in DEEP_RESEARCH_KEY_MAP


class TestOpenRouterModels:
    def test_fusion_is_offered(self):
        names = [m.name for m in OPENROUTER_MODELS]
        assert "openrouter/fusion" in names

    def test_all_openrouter_models_share_source_and_category(self):
        for model in OPENROUTER_MODELS:
            assert model.source == "openrouter"
            assert model.category == "openrouter"

    def test_openrouter_key_is_mapped(self):
        assert DEEP_RESEARCH_KEY_MAP["openrouter"] == "OPENROUTER_API_KEY"

    def test_discover_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert discover_openrouter_models() == []

    def test_discover_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        result = discover_openrouter_models()
        assert len(result) == len(OPENROUTER_MODELS)
        assert "openrouter/fusion" in [m.name for m in result]


class TestLlmModels:
    def test_discover_without_llm_installed(self, monkeypatch):
        """Should return empty list if llm import fails."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "llm":
                raise ImportError("no llm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = discover_llm_models()
        assert result == []


class TestCatalogueMatchesProviders:
    """Every model named in the docs must be discoverable, not just accepted."""

    def test_both_deepseek_tiers_are_discoverable(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        names = {m.name for m in discover_deep_research_models() if m.source == "deepseek"}
        assert names == {"deepseek-v4-flash", "deepseek-v4-pro"}

    def test_no_retired_model_names_in_catalogue(self):
        retired = {"deepseek-reasoner", "deepseek-chat", "grok-4.1-fast"}
        assert not {m.name for m in DEEP_RESEARCH_MODELS} & retired
