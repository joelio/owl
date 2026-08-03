"""Tests for terminal output formatting."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from owl.output import print_responses
from owl.providers.base import OwlResponse


def _capture(responses: list[OwlResponse]) -> str:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True, width=120)
    print_responses(responses, console)
    return output.getvalue()


class TestPrintResponses:
    def test_empty_responses(self):
        text = _capture([])
        assert "0 of 0" in text

    def test_successful_responses(self):
        responses = [
            OwlResponse(model_name="gpt-5", source="llm", text="Answer one"),
            OwlResponse(model_name="claude", source="llm", text="Answer two"),
        ]
        text = _capture(responses)
        assert "2 of 2" in text
        assert "gpt-5" in text
        assert "claude" in text

    def test_mixed_success_and_error(self):
        responses = [
            OwlResponse(model_name="gpt-5", source="llm", text="Good answer"),
            OwlResponse(model_name="broken", source="llm", text="", error="API failed"),
        ]
        text = _capture(responses)
        assert "1 of 2" in text
        assert "ERROR" in text

    def test_response_with_citations(self):
        responses = [
            OwlResponse(
                model_name="sonar",
                source="perplexity",
                text="Answer",
                citations=["https://example.com"],
            ),
        ]
        text = _capture(responses)
        assert "Sources" in text

    def test_timing_badge_shown(self):
        responses = [
            OwlResponse(model_name="gpt-5", source="llm", text="Answer", elapsed_seconds=3.2),
        ]
        text = _capture(responses)
        assert "3.2s" in text

    def test_timing_badge_absent_when_none(self):
        responses = [
            OwlResponse(model_name="gpt-5", source="llm", text="Answer"),
        ]
        text = _capture(responses)
        # Should not contain a timing badge like "(Nones)"
        assert "None" not in text


class TestMarkupIsNotLeakedToUsers:
    """Console markup tags must never reach the terminal as literal text.

    Markdown() does not parse console markup, so labels built as
    "[dim]Reasoning:[/dim]" used to be printed verbatim.
    """

    def test_reasoning_label_has_no_markup_tags(self):
        text = _capture(
            [
                OwlResponse(
                    model_name="deepseek-reasoner",
                    source="deepseek",
                    text="The answer",
                    reasoning="Step one, step two",
                )
            ]
        )
        assert "Reasoning" in text
        assert "[dim]" not in text
        assert "[/dim]" not in text

    def test_sources_label_has_no_markup_tags(self):
        text = _capture(
            [
                OwlResponse(
                    model_name="sonar",
                    source="perplexity",
                    text="The answer",
                    citations=["https://example.test/a"],
                )
            ]
        )
        assert "Sources" in text
        assert "[bold]" not in text
        assert "[/bold]" not in text
        assert "https://example.test/a" in text


class TestErrorTextIsPreserved:
    """Errors carry API payload snippets, which contain brackets."""

    def test_bracketed_error_text_survives(self):
        text = _capture(
            [
                OwlResponse(
                    model_name="broken",
                    source="llm",
                    text="",
                    error="unexpected response: see [docs] item [1]",
                )
            ]
        )
        assert "[docs]" in text
        assert "[1]" in text

    def test_unclosed_bracket_does_not_raise(self):
        # rich raises MarkupError on some malformed tags when markup is parsed.
        text = _capture(
            [
                OwlResponse(
                    model_name="broken",
                    source="llm",
                    text="",
                    error="bad payload: {'detail': '[unterminated'}",
                )
            ]
        )
        assert "unterminated" in text
