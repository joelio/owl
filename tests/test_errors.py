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
