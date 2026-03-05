"""Model discovery - finds available models from llm plugins and deep research APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AvailableModel:
    name: str
    source: str  # "llm", "openai-deep", "perplexity", "google-deep", "deepseek", "xai"
    category: str  # "standard" or "deep-research"
    description: str = ""


# Deep research models that are always available if the API key is set
DEEP_RESEARCH_MODELS = [
    AvailableModel(
        name="o3-deep-research",
        source="openai-deep",
        category="deep-research",
        description="OpenAI Deep Research (multi-step web synthesis)",
    ),
    AvailableModel(
        name="o4-mini-deep-research",
        source="openai-deep",
        category="deep-research",
        description="OpenAI Deep Research (faster, cheaper)",
    ),
    AvailableModel(
        name="sonar-deep-research",
        source="perplexity",
        category="deep-research",
        description="Perplexity Deep Research (web search + citations)",
    ),
    AvailableModel(
        name="gemini-deep-research",
        source="google-deep",
        category="deep-research",
        description="Gemini Deep Research Agent (async, Interactions API)",
    ),
    AvailableModel(
        name="deepseek-reasoner",
        source="deepseek",
        category="deep-research",
        description="DeepSeek Reasoner (chain-of-thought reasoning)",
    ),
    AvailableModel(
        name="grok-agentic",
        source="xai",
        category="deep-research",
        description="Grok 4.1 agentic search + thinking mode",
    ),
]

# Map of deep research source -> required env var
DEEP_RESEARCH_KEY_MAP = {
    "openai-deep": "OPENAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
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


def discover_deep_research_models() -> list[AvailableModel]:
    """Return deep research models that have API keys configured."""
    available = []
    for model in DEEP_RESEARCH_MODELS:
        env_var = DEEP_RESEARCH_KEY_MAP.get(model.source, "")
        if os.environ.get(env_var):
            available.append(model)
    return available


def discover_all_models() -> list[AvailableModel]:
    """Discover all available models from llm and deep research APIs."""
    return discover_llm_models() + discover_deep_research_models()
