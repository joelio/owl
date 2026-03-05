"""Tests for terminal output formatting."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from owl.output import print_responses
from owl.providers.base import OwlResponse


class TestPrintResponses:
    def _capture(self, responses: list[OwlResponse]) -> str:
        output = StringIO()
        console = Console(file=output, force_terminal=False, no_color=True, width=120)
        print_responses(responses, console)
        return output.getvalue()

    def test_empty_responses(self):
        text = self._capture([])
        assert "0 of 0" in text

    def test_successful_responses(self):
        responses = [
            OwlResponse(model_name="gpt-5", source="llm", text="Answer one"),
            OwlResponse(model_name="claude", source="llm", text="Answer two"),
        ]
        text = self._capture(responses)
        assert "2 of 2" in text
        assert "gpt-5" in text
        assert "claude" in text

    def test_mixed_success_and_error(self):
        responses = [
            OwlResponse(model_name="gpt-5", source="llm", text="Good answer"),
            OwlResponse(model_name="broken", source="llm", text="", error="API failed"),
        ]
        text = self._capture(responses)
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
        text = self._capture(responses)
        assert "Sources" in text
