"""Tests for the council picker.

The selection state is tested directly; the loop is driven by feeding
scripted lines to input().
"""

from __future__ import annotations

import pytest

from owl.config import Config, CouncilMember
from owl.models import AvailableModel
from owl.tui import MAX_VISIBLE, Selection, run_council_selector


def _model(name: str, source: str = "llm", category: str = "standard") -> AvailableModel:
    return AvailableModel(name=name, source=source, category=category, description="")


MODELS = [
    _model("gpt-5"),
    _model("claude-sonnet-4.6"),
    _model("llama-3.3-70b:free"),
    _model("o3-deep-research", source="openai-deep", category="deep-research"),
]


class TestSelection:
    def test_starts_from_the_configured_council(self):
        config = Config(council=[CouncilMember(name="gpt-5", source="llm")])
        selection = Selection.from_config(MODELS, config)
        assert selection.is_selected(MODELS[0])
        assert not selection.is_selected(MODELS[1])

    def test_ignores_configured_models_that_no_longer_exist(self):
        config = Config(council=[CouncilMember(name="retired-model", source="llm")])
        assert Selection.from_config(MODELS, config).selected == set()

    def test_toggle_selects_then_deselects(self):
        selection = Selection(models=MODELS)
        assert selection.toggle(1)
        assert selection.is_selected(MODELS[0])
        assert selection.toggle(1)
        assert not selection.is_selected(MODELS[0])

    @pytest.mark.parametrize("position", [0, -1, 99])
    def test_toggle_rejects_out_of_range(self, position):
        selection = Selection(models=MODELS)
        assert not selection.toggle(position)
        assert selection.selected == set()

    def test_same_name_different_source_are_distinct(self):
        models = [_model("grok", source="llm"), _model("grok", source="xai")]
        selection = Selection(models=models)
        selection.toggle(1)
        assert selection.is_selected(models[0])
        assert not selection.is_selected(models[1])


class TestFilter:
    def test_matches_on_name(self):
        selection = Selection(models=MODELS, query="claude")
        assert [m.name for m in selection.visible()] == ["claude-sonnet-4.6"]

    def test_matches_on_source(self):
        selection = Selection(models=MODELS, query="openai-deep")
        assert [m.name for m in selection.visible()] == ["o3-deep-research"]

    def test_is_case_insensitive(self):
        assert Selection(models=MODELS, query="CLAUDE").visible()

    def test_empty_query_shows_everything(self):
        assert len(Selection(models=MODELS, query="").visible()) == len(MODELS)

    def test_toggle_position_is_relative_to_the_filter(self):
        selection = Selection(models=MODELS, query="claude")
        selection.toggle(1)
        # Position 1 of the filtered view, not of the full catalogue.
        assert selection.is_selected(MODELS[1])
        assert not selection.is_selected(MODELS[0])

    def test_select_visible_only_touches_matches(self):
        selection = Selection(models=MODELS, query=":free")
        selection.select_visible()
        assert selection.council() == [CouncilMember(name="llama-3.3-70b:free", source="llm")]

    def test_clear_visible_leaves_hidden_selections_alone(self):
        selection = Selection(models=MODELS)
        selection.select_visible()
        selection.query = "claude"
        selection.clear_visible()
        names = {m.name for m in selection.council()}
        assert "claude-sonnet-4.6" not in names
        assert "gpt-5" in names


class TestLargeCatalogues:
    """OpenRouter alone puts several hundred models in the list."""

    def _many(self, count: int) -> list[AvailableModel]:
        return [_model(f"model-{i:03d}") for i in range(count)]

    def test_display_is_capped(self):
        selection = Selection(models=self._many(300))
        assert len(selection.visible()) == MAX_VISIBLE

    def test_hidden_count_reports_the_remainder(self):
        selection = Selection(models=self._many(300))
        assert selection.hidden_count() == 300 - MAX_VISIBLE

    def test_filtering_reaches_past_the_cap(self):
        selection = Selection(models=self._many(300), query="model-287")
        assert [m.name for m in selection.visible()] == ["model-287"]

    def test_no_hidden_count_when_everything_fits(self):
        assert Selection(models=self._many(5)).hidden_count() == 0

    def test_council_order_follows_the_catalogue_not_selection_order(self):
        selection = Selection(models=MODELS)
        selection.toggle(3)
        selection.toggle(1)
        assert [m.name for m in selection.council()] == ["gpt-5", "llama-3.3-70b:free"]


class TestSelectorLoop:
    @pytest.fixture
    def wired(self, monkeypatch, tmp_path):
        saved = {}
        monkeypatch.setattr("owl.tui.discover_all_models", lambda: list(MODELS))
        monkeypatch.setattr("owl.tui.load_config", lambda: Config())
        monkeypatch.setattr("owl.tui.save_config", lambda c: saved.update(council=c.council))
        return saved

    def _feed(self, monkeypatch, lines):
        it = iter(lines)
        monkeypatch.setattr("builtins.input", lambda: next(it))

    def test_ctrl_d_cancels_without_saving(self, monkeypatch, wired):
        def eof():
            raise EOFError

        monkeypatch.setattr("builtins.input", eof)
        result = run_council_selector()
        assert result.council == []
        assert wired == {}

    def test_ctrl_c_cancels_without_saving(self, monkeypatch, wired):
        def interrupt():
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupt)
        run_council_selector()
        assert wired == {}

    def test_quit_does_not_save(self, monkeypatch, wired):
        self._feed(monkeypatch, ["1", "q"])
        run_council_selector()
        assert wired == {}

    def test_toggle_then_save(self, monkeypatch, wired):
        self._feed(monkeypatch, ["1", "s"])
        run_council_selector()
        assert wired["council"] == [CouncilMember(name="gpt-5", source="llm")]

    def test_filter_then_select_all_shown(self, monkeypatch, wired):
        self._feed(monkeypatch, ["/claude", "a", "s"])
        run_council_selector()
        assert wired["council"] == [CouncilMember(name="claude-sonnet-4.6", source="llm")]

    def test_slash_alone_clears_the_filter(self, monkeypatch, wired):
        self._feed(monkeypatch, ["/claude", "/", "a", "s"])
        run_council_selector()
        assert len(wired["council"]) == len(MODELS)

    def test_out_of_range_number_is_survivable(self, monkeypatch, wired):
        self._feed(monkeypatch, ["99", "1", "s"])
        run_council_selector()
        assert wired["council"] == [CouncilMember(name="gpt-5", source="llm")]

    def test_no_models_returns_config_untouched(self, monkeypatch):
        monkeypatch.setattr("owl.tui.discover_all_models", list)
        monkeypatch.setattr("owl.tui.load_config", lambda: Config())
        assert run_council_selector().council == []
