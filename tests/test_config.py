"""Tests for config management."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from owl.config import Config, CouncilMember, load_config, save_config


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Use a temp directory for config."""
    monkeypatch.setattr("owl.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("owl.config.CONFIG_FILE", tmp_path / "config.yaml")
    return tmp_path


class TestCouncilMember:
    def test_to_dict(self):
        m = CouncilMember(name="gpt-5", source="llm")
        assert m.to_dict() == {"name": "gpt-5", "source": "llm"}

    def test_from_dict(self):
        m = CouncilMember.from_dict({"name": "sonar-deep-research", "source": "perplexity"})
        assert m.name == "sonar-deep-research"
        assert m.source == "perplexity"


class TestConfig:
    def test_empty_config(self):
        c = Config()
        assert c.council == []
        assert c.to_dict() == {"council": []}

    def test_roundtrip(self):
        members = [
            CouncilMember("gpt-5", "llm"),
            CouncilMember("o3-deep-research", "openai-deep"),
        ]
        c = Config(council=members)
        d = c.to_dict()
        c2 = Config.from_dict(d)
        assert len(c2.council) == 2
        assert c2.council[0].name == "gpt-5"
        assert c2.council[1].source == "openai-deep"

    def test_from_dict_missing_council(self):
        c = Config.from_dict({})
        assert c.council == []


class TestLoadSave:
    def test_load_missing_file(self, config_dir):
        config = load_config()
        assert config.council == []

    def test_save_and_load(self, config_dir):
        config = Config(council=[CouncilMember("claude-sonnet", "llm")])
        save_config(config)

        loaded = load_config()
        assert len(loaded.council) == 1
        assert loaded.council[0].name == "claude-sonnet"

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "nested"
        monkeypatch.setattr("owl.config.CONFIG_DIR", nested)
        monkeypatch.setattr("owl.config.CONFIG_FILE", nested / "config.yaml")

        save_config(Config(council=[CouncilMember("test", "llm")]))
        assert (nested / "config.yaml").exists()

    def test_config_file_is_valid_yaml(self, config_dir):
        config = Config(
            council=[
                CouncilMember("gpt-5", "llm"),
                CouncilMember("sonar-deep-research", "perplexity"),
            ]
        )
        save_config(config)

        with open(config_dir / "config.yaml") as f:
            data = yaml.safe_load(f)
        assert len(data["council"]) == 2
