"""Tests for retry logic.

Sleeps are patched out rather than waited on: these assertions are about
what gets retried and for how long, and real backoff made the suite spend
16 of its 17 seconds asleep.
"""

from __future__ import annotations

import httpx
import pytest

from owl.providers import retry as retry_module
from owl.providers.retry import MAX_RETRY_AFTER_SECONDS, parse_retry_after, with_retry


@pytest.fixture
def slept(monkeypatch):
    """Record requested sleeps instead of performing them."""
    recorded: list[float] = []

    async def fake_sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)
    # Remove jitter so delays are exact and assertions stay readable.
    monkeypatch.setattr(retry_module, "JITTER_RATIO", 0.0)
    return recorded


def _status_error(code: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(code, request=request, headers=headers or {})
    return httpx.HTTPStatusError("boom", request=request, response=response)


class TestRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self, slept):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            return "ok"

        assert await with_retry(fn) == "ok"
        assert calls == 1
        assert slept == []

    @pytest.mark.asyncio
    async def test_retries_on_429(self, slept):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _status_error(429)
            return "ok"

        assert await with_retry(fn) == "ok"
        assert calls == 3
        assert slept == [2.0, 5.0]

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, slept):
        async def fn():
            raise _status_error(429)

        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(fn)

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self, slept):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise _status_error(400)

        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(fn)
        assert calls == 1
        assert slept == []

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self, slept):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise httpx.TimeoutException("timed out")
            return "ok"

        assert await with_retry(fn) == "ok"
        assert calls == 2

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_retries_every_transient_status(self, slept, code):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise _status_error(code)
            return "ok"

        assert await with_retry(fn) == "ok"
        assert calls == 2


class TestTransportErrors:
    """A dropped connection is as transient as a 503."""

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("refused"),
            httpx.ReadError("reset"),
            httpx.WriteError("broken pipe"),
            httpx.RemoteProtocolError("bad chunk"),
        ],
    )
    @pytest.mark.asyncio
    async def test_transport_errors_are_retried(self, slept, exc):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise exc
            return "ok"

        assert await with_retry(fn) == "ok"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_non_transient_errors_are_not_retried(self, slept):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise ValueError("programming error")

        with pytest.raises(ValueError):
            await with_retry(fn)
        assert calls == 1


class TestRetryAfter:
    """A server saying when to come back beats our guess."""

    def test_parses_seconds(self):
        assert parse_retry_after("12") == 12.0

    def test_ignores_missing_header(self):
        assert parse_retry_after(None) is None

    def test_ignores_unparseable(self):
        assert parse_retry_after("soon please") is None

    def test_ignores_zero_and_negative(self):
        assert parse_retry_after("0") is None
        assert parse_retry_after("-5") is None

    def test_ignores_absurdly_long_waits(self):
        assert parse_retry_after(str(MAX_RETRY_AFTER_SECONDS + 1)) is None

    def test_parses_http_date(self):
        import datetime as dt
        from email.utils import format_datetime

        soon = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)
        parsed = parse_retry_after(format_datetime(soon))
        assert parsed is not None
        assert 25 < parsed <= 31

    @pytest.mark.asyncio
    async def test_header_overrides_the_backoff_table(self, slept):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise _status_error(429, {"Retry-After": "7"})
            return "ok"

        assert await with_retry(fn) == "ok"
        assert slept == [7.0]

    @pytest.mark.asyncio
    async def test_falls_back_when_header_is_absurd(self, slept):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise _status_error(429, {"Retry-After": "86400"})
            return "ok"

        assert await with_retry(fn) == "ok"
        assert slept == [2.0]  # the table, not a day


class TestJitter:
    @pytest.mark.asyncio
    async def test_delay_is_jittered_within_bounds(self, monkeypatch):
        recorded: list[float] = []

        async def fake_sleep(seconds):
            recorded.append(seconds)

        monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)

        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise _status_error(429)
            return "ok"

        await with_retry(fn)
        # Base 2.0 plus up to 25%, so never below base and never far above.
        assert 2.0 <= recorded[0] <= 2.5


class TestRetryAfterEdgeCases:
    """Values that slip past a naive bounds check."""

    def test_nan_is_rejected(self):
        # NaN passes both `<= 0` and `> MAX`, and asyncio.sleep(nan) never wakes.
        assert parse_retry_after("nan") is None

    @pytest.mark.parametrize("value", ["inf", "-inf", "infinity"])
    def test_infinities_are_rejected(self, value):
        assert parse_retry_after(value) is None

    def test_naive_date_is_read_as_utc(self):
        """A "-0000" date means UTC, not local time."""
        import datetime as dt

        soon = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)
        # "-0000" makes parsedate_to_datetime return a naive datetime.
        header = soon.strftime("%a, %d %b %Y %H:%M:%S -0000")
        parsed = parse_retry_after(header)
        assert parsed is not None
        assert 25 < parsed <= 31


class TestLogSafety:
    def test_newlines_cannot_forge_a_log_entry(self):
        from owl.providers.retry import _log_safe

        out = _log_safe("https://x.test/v1/a\r\nINFO forged entry")
        assert "\n" not in out
        assert "\r" not in out

    def test_credentials_are_still_redacted(self):
        from owl.providers.retry import _log_safe

        assert "SECRET" not in _log_safe("https://x.test/?key=SECRET")
