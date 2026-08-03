"""Error text that is safe to show a user and safe to post to GitHub.

Provider errors travel further than most: ``output.py`` prints them to the
terminal and ``github.py`` writes them into a comment that may land on a
public issue.  Anything derived from an exception therefore has to have
credentials stripped out of it first.

The specific hazard is APIs that take the key as a query parameter.  httpx
puts the full request URL into ``str(HTTPStatusError)``, so a plain
``str(exc)`` on a failed request carries the key with it.
"""

from __future__ import annotations

import re

import httpx

REDACTED = "REDACTED"

# Query parameters whose values are credentials.  Google's Generative
# Language API is the one owl actually calls this way, but the others are
# common enough in redirect and proxy URLs to be worth covering.
_CREDENTIAL_QUERY_PARAMS = (
    "key",
    "api_key",
    "apikey",
    "access_token",
    "token",
    "auth",
)

_CREDENTIAL_RE = re.compile(
    r"([?&](?:" + "|".join(_CREDENTIAL_QUERY_PARAMS) + r")=)([^&\s'\"<>]+)",
    re.IGNORECASE,
)


def redact(text: str) -> str:
    """Mask the values of credential-bearing query parameters in ``text``.

    Works on arbitrary text rather than parsed URLs, because the URL is
    usually already embedded in an exception message by the time we see it.
    """
    return _CREDENTIAL_RE.sub(rf"\g<1>{REDACTED}", text)


_MAX_DETAIL_CHARS = 300


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _MAX_DETAIL_CHARS:
        return text
    return text[:_MAX_DETAIL_CHARS] + "..."


def response_detail(response: httpx.Response) -> str:
    """Pull the human-readable reason out of an error response body.

    The status line alone says a request failed but not why. Every provider
    owl calls returns the reason in the body, and all of them use some
    variant of ``{"error": {"message": ...}}``, so try that shape first and
    fall back to the raw body.
    """
    try:
        body = response.text
    except Exception:  # body was never read, e.g. a streamed response
        return ""

    if not body.strip():
        return ""

    try:
        data = response.json()
    except ValueError:
        return _truncate(body)

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return _truncate(str(error["message"]))
        if isinstance(error, str) and error:
            return _truncate(error)
        if data.get("message"):
            return _truncate(str(data["message"]))

    return _truncate(body)


def describe_error(exc: BaseException) -> str:
    """Return a short, credential-free description of a provider failure."""
    if isinstance(exc, httpx.HTTPStatusError):
        summary = f"HTTP {exc.response.status_code} from {exc.request.url}"
        detail = response_detail(exc.response)
        return redact(f"{summary}: {detail}" if detail else summary)

    message = redact(str(exc)).strip()
    return message or exc.__class__.__name__
