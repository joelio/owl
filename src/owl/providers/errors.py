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


def describe_error(exc: BaseException) -> str:
    """Return a short, credential-free description of a provider failure."""
    if isinstance(exc, httpx.HTTPStatusError):
        return redact(f"HTTP {exc.response.status_code} from {exc.request.url}")

    message = redact(str(exc)).strip()
    return message or exc.__class__.__name__
