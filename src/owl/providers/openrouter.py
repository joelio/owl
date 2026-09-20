"""OpenRouter provider — one standard endpoint, any model slug.

This is the general-purpose route for council members.  A member is declared
with the OpenRouter slug as its ``name`` (e.g. ``anthropic/claude-sonnet-4.6``
or ``openrouter/fusion``) and ``source: openrouter``; the slug is sent through
unchanged and the reply is attributed back to that exact slug.
"""

from __future__ import annotations

from typing import Any

from .http_base import HttpProvider, build_messages, parse_chat_message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Sent so owl shows up in OpenRouter's app leaderboard rather than as "unknown".
_REFERER = "https://github.com/joelio/owl"
_TITLE = "Parliament of Owls"


def _extract_citations(message: dict[str, Any]) -> list[str] | None:
    """Pull URLs out of OpenRouter's ``annotations`` array, if present.

    Models with web search enabled (Fusion, the ``:online`` variants, Perplexity
    Sonar via OpenRouter) return their sources here.  Models without search
    return nothing, in which case there are no citations to show.
    """
    annotations = message.get("annotations")
    if not isinstance(annotations, list):
        return None

    urls: list[str] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        citation = annotation.get("url_citation")
        if isinstance(citation, dict) and citation.get("url"):
            url = citation["url"]
            if url not in urls:
                urls.append(url)
    return urls or None


class OpenRouterProvider(HttpProvider):
    source = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def build_request(self, prompt: str, system_prompt: str | None, api_key: str) -> dict[str, Any]:
        return {
            "url": f"{OPENROUTER_BASE_URL}/chat/completions",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": _REFERER,
                "X-Title": _TITLE,
            },
            "json": {
                "model": self.model_name,
                "messages": build_messages(prompt, system_prompt),
            },
        }

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        message = parse_chat_message(data)
        return {
            "text": message["content"],
            "citations": _extract_citations(message),
            "reasoning": message.get("reasoning"),
        }
