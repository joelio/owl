"""Tests for model discovery."""

from __future__ import annotations

import pytest

from owl.models import (
    DEEP_RESEARCH_KEY_MAP,
    DEEP_RESEARCH_MODELS,
    AvailableModel,
    discover_deep_research_models,
    discover_llm_models,
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

    def test_discover_with_openai_key(self, monkeypatch):
        for env_var in DEEP_RESEARCH_KEY_MAP.values():
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        result = discover_deep_research_models()
        names = [m.name for m in result]
        assert "o3-deep-research" in names
        assert "o4-mini-deep-research" in names
        assert "sonar-deep-research" not in names

    def test_discover_with_all_keys(self, monkeypatch):
        for env_var in DEEP_RESEARCH_KEY_MAP.values():
            monkeypatch.setenv(env_var, "test-key")
        result = discover_deep_research_models()
        assert len(result) == len(DEEP_RESEARCH_MODELS)

    def test_all_deep_models_are_deep_category(self):
        for model in DEEP_RESEARCH_MODELS:
            assert model.category == "deep-research"


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
