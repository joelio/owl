"""Tests for CLI commands."""

from __future__ import annotations


import pytest
from click.testing import CliRunner

from owl.cli.main import cli
from owl.config import Config, CouncilMember
from owl.providers.base import OwlResponse


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "owl" in result.output

    def test_ask_no_council(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr("owl.config.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("owl.config.CONFIG_FILE", tmp_path / "config.yaml")
        monkeypatch.setattr("owl.cli.main.load_config", lambda: Config())

        result = runner.invoke(cli, ["ask", "test prompt"])
        assert "No council members" in result.output

    def test_ask_with_council(self, runner, monkeypatch):
        config = Config(council=[CouncilMember("test-model", "llm")])
        monkeypatch.setattr("owl.cli.main.load_config", lambda: config)

        mock_responses = [OwlResponse(model_name="test-model", source="llm", text="Test answer")]

        async def mock_convene(*args, **kwargs):
            return mock_responses

        monkeypatch.setattr("owl.cli.main.convene", mock_convene)

        result = runner.invoke(cli, ["ask", "what is python"])
        assert result.exit_code == 0
        assert "test-model" in result.output

    def test_council_list_empty(self, runner, monkeypatch):
        monkeypatch.setattr("owl.cli.main.load_config", lambda: Config())
        result = runner.invoke(cli, ["council-list"])
        assert "No council members" in result.output

    def test_council_list_with_members(self, runner, monkeypatch):
        config = Config(
            council=[
                CouncilMember("gpt-5", "llm"),
                CouncilMember("sonar-deep-research", "perplexity"),
            ]
        )
        monkeypatch.setattr("owl.cli.main.load_config", lambda: config)
        result = runner.invoke(cli, ["council-list"])
        assert "gpt-5" in result.output
        assert "sonar-deep-research" in result.output

    def test_ask_with_format_flag(self, runner, monkeypatch):
        config = Config(council=[CouncilMember("test-model", "llm")])
        monkeypatch.setattr("owl.cli.main.load_config", lambda: config)

        captured_kwargs = {}

        async def mock_convene(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return [OwlResponse(model_name="test-model", source="llm", text="Answer")]

        monkeypatch.setattr("owl.cli.main.convene", mock_convene)

        result = runner.invoke(cli, ["ask", "--format", "brief", "test prompt"])
        assert result.exit_code == 0
        from owl.prompts import ResponseFormat

        assert captured_kwargs.get("fmt") == ResponseFormat.BRIEF

    def test_ask_format_default_is_standard(self, runner, monkeypatch):
        config = Config(council=[CouncilMember("test-model", "llm")])
        monkeypatch.setattr("owl.cli.main.load_config", lambda: config)

        captured_kwargs = {}

        async def mock_convene(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return [OwlResponse(model_name="test-model", source="llm", text="Answer")]

        monkeypatch.setattr("owl.cli.main.convene", mock_convene)

        result = runner.invoke(cli, ["ask", "test prompt"])
        assert result.exit_code == 0
        from owl.prompts import ResponseFormat

        assert captured_kwargs.get("fmt") == ResponseFormat.STANDARD

    def test_models_empty(self, runner, monkeypatch):
        monkeypatch.setattr("owl.cli.main.discover_all_models", lambda: [])
        result = runner.invoke(cli, ["models"])
        assert "No models found" in result.output
