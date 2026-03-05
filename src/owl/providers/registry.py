"""Provider registry - maps council members to their providers."""

from __future__ import annotations

from ..config import CouncilMember
from .base import Provider
from .deepseek import DeepSeekProvider
from .google_deep import GoogleDeepProvider
from .llm_provider import LlmProvider
from .openai_deep import OpenAIDeepProvider
from .perplexity import PerplexityProvider
from .xai import XAIProvider

SOURCE_TO_PROVIDER: dict[str, type[Provider]] = {
    "openai-deep": OpenAIDeepProvider,
    "perplexity": PerplexityProvider,
    "google-deep": GoogleDeepProvider,
    "deepseek": DeepSeekProvider,
    "xai": XAIProvider,
}


def get_provider(member: CouncilMember) -> Provider:
    """Get the appropriate provider for a council member."""
    if member.source == "llm":
        return LlmProvider(model_id=member.name)

    provider_cls = SOURCE_TO_PROVIDER.get(member.source)
    if provider_cls is None:
        raise ValueError(f"Unknown source: {member.source}")

    return provider_cls(model_name=member.name)
