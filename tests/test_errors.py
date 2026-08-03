"""Tests for credential-safe error text.

Provider errors are printed to the terminal and posted to GitHub issues, so
a leaked key here becomes a published key.
"""

from __future__ import annotations

import httpx
import pytest

from owl.providers.errors import describe_error, redact

SECRET = "AIzaSyTOTALLYREALKEY123"


class TestRedact:
    def test_masks_key_query_param(self):
        out = redact(f"https://example.test/v1/thing?key={SECRET}")
        assert SECRET not in out
        assert "key=REDACTED" in out

    @pytest.mark.parametrize("param", ["key", "api_key", "apikey", "access_token", "token", "auth"])
    def test_masks_each_credential_param(self, param):
        assert SECRET not in redact(f"https://example.test/?{param}={SECRET}")

    def test_masks_param_in_second_position(self):
        out = redact(f"https://example.test/?alt=json&key={SECRET}")
        assert SECRET not in out
        assert "alt=json" in out

    def test_is_case_insensitive(self):
        assert SECRET not in redact(f"https://example.test/?KEY={SECRET}")

    def test_stops_at_next_param(self):
        out = redact(f"https://example.test/?key={SECRET}&alt=json")
        assert SECRET not in out
        assert "alt=json" in out

    def test_leaves_ordinary_text_alone(self):
        text = "unexpected response (no choices): {'error': 'bad model'}"
        assert redact(text) == text


class TestDescribeError:
    def test_http_error_does_not_leak_key(self):
        request = httpx.Request("GET", f"https://example.test/v1/x?key={SECRET}")
        exc = httpx.HTTPStatusError("boom", request=request, response=httpx.Response(401))
        message = describe_error(exc)
        assert SECRET not in message
        assert "401" in message

    def test_generic_exception_is_redacted(self):
        exc = RuntimeError(f"failed calling https://example.test/?key={SECRET}")
        assert SECRET not in describe_error(exc)

    def test_generic_exception_keeps_message(self):
        assert describe_error(ValueError("unexpected response")) == "unexpected response"

    def test_empty_message_falls_back_to_type_name(self):
        assert describe_error(httpx.ReadTimeout("")) == "ReadTimeout"


class TestResponseDetail:
    """The status line says a request failed; the body says why."""

    def _error(self, status: int, **response_kwargs) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "https://example.test/v1/chat")
        response = httpx.Response(status, request=request, **response_kwargs)
        return httpx.HTTPStatusError("boom", request=request, response=response)

    def test_openai_style_error_message_is_surfaced(self):
        exc = self._error(
            400,
            json={"error": {"message": "Invalid model 'o3-deep-research'", "type": "invalid"}},
        )
        message = describe_error(exc)
        assert "400" in message
        assert "Invalid model 'o3-deep-research'" in message

    def test_top_level_message_is_surfaced(self):
        assert "quota exhausted" in describe_error(
            self._error(429, json={"message": "quota exhausted"})
        )

    def test_string_error_field_is_surfaced(self):
        assert "bad request" in describe_error(self._error(400, json={"error": "bad request"}))

    def test_non_json_body_falls_back_to_raw_text(self):
        assert "upstream timeout" in describe_error(self._error(504, text="upstream timeout"))

    def test_empty_body_gives_status_only(self):
        assert describe_error(self._error(503)) == "HTTP 503 from https://example.test/v1/chat"

    def test_long_body_is_truncated(self):
        message = describe_error(self._error(400, text="x" * 5000))
        assert message.endswith("...")
        assert len(message) < 500

    def test_detail_is_redacted(self):
        exc = self._error(401, json={"error": {"message": f"bad key at ?key={SECRET}"}})
        assert SECRET not in describe_error(exc)


class TestLoggingIsQuietByDefault:
    """Provider failures are reported as error panels.

    Python's logging.lastResort handler prints WARNING-and-above records,
    tracebacks included, whenever no handler is configured. The package
    installs a NullHandler so those tracebacks stay behind `owl -v`.
    """

    def test_owl_logger_has_a_handler(self):
        import logging

        handlers = logging.getLogger("owl").handlers
        assert any(isinstance(h, logging.NullHandler) for h in handlers)
