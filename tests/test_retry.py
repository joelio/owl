"""Tests for retry logic."""

from __future__ import annotations

import pytest

from owl.providers.retry import with_retry


class TestRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(fn)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        import httpx

        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
                raise httpx.HTTPStatusError(
                    "rate limited", request=response.request, response=response
                )
            return "ok"

        result = await with_retry(fn)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        import httpx

        async def fn():
            response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
            raise httpx.HTTPStatusError("rate limited", request=response.request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(fn)

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        import httpx

        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            response = httpx.Response(400, request=httpx.Request("POST", "http://test"))
            raise httpx.HTTPStatusError("bad request", request=response.request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(fn)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        import httpx

        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("timed out")
            return "ok"

        result = await with_retry(fn)
        assert result == "ok"
        assert call_count == 2
