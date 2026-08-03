"""Tests for the arbiter synthesis pass."""

from __future__ import annotations

import pytest

from owl.config import Config, CouncilMember
from owl.providers.base import OwlResponse
from owl.synthesis import SYNTHESIS_SOURCE, build_arbiter_input, synthesise

ARBITER = CouncilMember(name="judge-model", source="llm")


def _answer(name: str, text: str, **kwargs) -> OwlResponse:
    return OwlResponse(model_name=name, source="llm", text=text, **kwargs)


class TestBuildArbiterInput:
    def test_includes_the_original_question(self):
        text = build_arbiter_input("Redis or Memcached?", [_answer("a", "Redis")])
        assert "Redis or Memcached?" in text

    def test_includes_every_answer_in_full(self):
        text = build_arbiter_input("q", [_answer("a", "first answer"), _answer("b", "second")])
        assert "first answer" in text
        assert "second" in text

    def test_labels_each_member(self):
        text = build_arbiter_input("q", [_answer("gpt-5", "x"), _answer("claude", "y")])
        assert "gpt-5" in text
        assert "claude" in text

    def test_includes_reasoning_when_present(self):
        text = build_arbiter_input("q", [_answer("a", "answer", reasoning="because of X")])
        assert "because of X" in text

    def test_truncates_very_long_answers(self):
        text = build_arbiter_input("q", [_answer("a", "x" * 50_000)])
        assert "truncated for the arbiter" in text
        assert len(text) < 20_000


class TestSynthesise:
    @pytest.mark.asyncio
    async def test_skips_when_only_one_answer(self, monkeypatch):
        called = False

        def _fail(*a, **k):
            nonlocal called
            called = True
            raise AssertionError("arbiter should not be called")

        monkeypatch.setattr("owl.synthesis.get_provider", _fail)
        result = await synthesise("q", [_answer("a", "only one")], ARBITER)
        assert result is None
        assert not called

    @pytest.mark.asyncio
    async def test_skips_when_no_answers(self, monkeypatch):
        monkeypatch.setattr("owl.synthesis.get_provider", lambda m: None)
        responses = [OwlResponse(model_name="a", source="llm", text="", error="failed")]
        assert await synthesise("q", responses, ARBITER) is None

    @pytest.mark.asyncio
    async def test_ignores_failed_members(self, monkeypatch):
        """Two responses but one errored is still only one usable answer."""
        monkeypatch.setattr("owl.synthesis.get_provider", lambda m: None)
        responses = [
            _answer("a", "good"),
            OwlResponse(model_name="b", source="llm", text="", error="boom"),
        ]
        assert await synthesise("q", responses, ARBITER) is None

    @pytest.mark.asyncio
    async def test_returns_synthesis_for_two_answers(self, monkeypatch):
        seen = {}

        class FakeProvider:
            async def query(self, prompt, system_prompt=None):
                seen["prompt"] = prompt
                seen["system"] = system_prompt
                return OwlResponse(
                    model_name="judge-model", source="llm", text="ANSWER: Redis.", elapsed_seconds=1
                )

        monkeypatch.setattr("owl.synthesis.get_provider", lambda m: FakeProvider())
        result = await synthesise("q", [_answer("a", "Redis"), _answer("b", "Memcached")], ARBITER)

        assert result is not None
        assert result.text == "ANSWER: Redis."
        assert result.source == SYNTHESIS_SOURCE
        assert result.model_name == "judge-model"
        # The arbiter must see the answers, and be told to weigh reasoning.
        assert "Redis" in seen["prompt"]
        assert "Memcached" in seen["prompt"]
        assert "arbiter" in seen["system"].lower()

    @pytest.mark.asyncio
    async def test_arbiter_failure_becomes_an_error_response(self, monkeypatch):
        class Boom:
            async def query(self, prompt, system_prompt=None):
                raise RuntimeError("arbiter exploded")

        monkeypatch.setattr("owl.synthesis.get_provider", lambda m: Boom())
        result = await synthesise("q", [_answer("a", "x"), _answer("b", "y")], ARBITER)

        assert result is not None
        assert result.text == ""
        assert "arbiter exploded" in (result.error or "")


class TestResolveArbiter:
    def test_prefers_the_configured_arbiter(self):
        config = Config(
            council=[CouncilMember(name="gpt-5", source="llm")],
            arbiter=CouncilMember(name="chosen", source="llm"),
        )
        assert config.resolve_arbiter().name == "chosen"

    def test_falls_back_to_first_llm_member(self):
        config = Config(
            council=[
                CouncilMember(name="o3-deep-research", source="openai-deep"),
                CouncilMember(name="gpt-5", source="llm"),
            ]
        )
        # Deep research members are skipped: slow and costly for a
        # reconciling job.
        assert config.resolve_arbiter().name == "gpt-5"

    def test_returns_none_when_no_llm_member(self):
        config = Config(council=[CouncilMember(name="o3-deep-research", source="openai-deep")])
        assert config.resolve_arbiter() is None

    def test_arbiter_survives_a_config_roundtrip(self):
        config = Config(
            council=[CouncilMember(name="gpt-5", source="llm")],
            arbiter=CouncilMember(name="judge", source="llm"),
        )
        restored = Config.from_dict(config.to_dict())
        assert restored.arbiter == CouncilMember(name="judge", source="llm")

    def test_config_without_arbiter_roundtrips_to_none(self):
        config = Config(council=[CouncilMember(name="gpt-5", source="llm")])
        assert "arbiter" not in config.to_dict()
        assert Config.from_dict(config.to_dict()).arbiter is None
