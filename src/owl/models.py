"""Model discovery - finds available models from llm plugins and direct APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AvailableModel:
    name: str
    source: str  # "llm", "openrouter", "google-deep", "deepseek", "xai"
    category: str  # "standard", "openrouter" or "deep-research"
    description: str = ""


# OpenRouter models that are always available if OPENROUTER_API_KEY is set.
# These are a starting point, not a whitelist: any OpenRouter slug can be used
# as a council member by adding it to ~/.owl/config.yaml with source "openrouter".
# Slugs below were checked against https://openrouter.ai/api/v1/models on 2026-09-20.
OPENROUTER_MODELS = [
    AvailableModel(
        name="openrouter/fusion",
        source="openrouter",
        category="openrouter",
        description="OpenRouter Fusion (multi-model panel + web search, blended answer)",
    ),
    AvailableModel(
        name="perplexity/sonar-reasoning-pro",
        source="openrouter",
        category="openrouter",
        description="Perplexity Sonar Reasoning Pro (web search + citations)",
    ),
    AvailableModel(
        name="openai/gpt-5",
        source="openrouter",
        category="openrouter",
        description="OpenAI GPT-5",
    ),
    AvailableModel(
        name="anthropic/claude-sonnet-4.6",
        source="openrouter",
        category="openrouter",
        description="Anthropic Claude Sonnet 4.6",
    ),
    AvailableModel(
        name="google/gemini-3.5-flash",
        source="openrouter",
        category="openrouter",
        description="Google Gemini 3.5 Flash",
    ),
]

# Deep research models that are always available if the API key is set
DEEP_RESEARCH_MODELS = [
    AvailableModel(
        name="gemini-deep-research",
        source="google-deep",
        category="deep-research",
        description="Gemini Deep Research agent (Interactions API, polled)",
    ),
    AvailableModel(
        name="deepseek-v4-flash",
        source="deepseek",
        category="deep-research",
        description="DeepSeek V4 Flash (thinking mode)",
    ),
    AvailableModel(
        name="deepseek-v4-pro",
        source="deepseek",
        category="deep-research",
        description="DeepSeek V4 Pro (frontier tier, thinking mode)",
    ),
    AvailableModel(
        name="grok-agentic",
        source="xai",
        category="deep-research",
        description="Grok 4.5 agentic web + X search",
    ),
]

# Map of direct-API source -> required env var
DEEP_RESEARCH_KEY_MAP = {
    "openrouter": "OPENROUTER_API_KEY",
    "google-deep": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}


def discover_llm_models() -> list[AvailableModel]:
    """Discover models available through installed llm plugins."""
    try:
        import llm

        models = []
        for model in llm.get_models():
            models.append(
                AvailableModel(
                    name=model.model_id,
                    source="llm",
                    category="standard",
                    description=str(getattr(model, "description", "")),
                )
            )
        return models
    except Exception:
        return []


def _with_key(models: list[AvailableModel]) -> list[AvailableModel]:
    """Filter to models whose source has its API key set."""
    return [m for m in models if os.environ.get(DEEP_RESEARCH_KEY_MAP.get(m.source, ""))]


def discover_openrouter_models() -> list[AvailableModel]:
    """Return the suggested OpenRouter models, if OPENROUTER_API_KEY is set."""
    return _with_key(OPENROUTER_MODELS)


def discover_deep_research_models() -> list[AvailableModel]:
    """Return deep research models that have API keys configured."""
    return _with_key(DEEP_RESEARCH_MODELS)


def discover_all_models() -> list[AvailableModel]:
    """Discover all available models from llm plugins and direct APIs."""
    return discover_llm_models() + discover_openrouter_models() + discover_deep_research_models()
