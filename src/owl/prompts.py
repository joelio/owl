"""System prompt templates for optimising response quality across providers."""

from __future__ import annotations

from enum import Enum


class ResponseFormat(Enum):
    """Controls target response length."""

    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


# Word-count targets per format level
_WORD_TARGETS: dict[ResponseFormat, tuple[int, int]] = {
    ResponseFormat.BRIEF: (100, 200),
    ResponseFormat.STANDARD: (250, 400),
    ResponseFormat.DETAILED: (600, 1000),
}

_STRUCTURE_BLOCK = """\
Your response MUST use exactly this structure:

ANSWER: [Direct answer in 1-3 sentences]

REASONING: [Core evidence or reasoning, 2-4 sentences]

CAVEATS: [Uncertainties or limits on your answer. Write "None." if none apply.]

KEY POINTS:
- [Point 1]
- [Point 2]
- [Point 3, maximum]"""

_UNCERTAINTY_RULE = (
    "When you are uncertain, say so explicitly. "
    'Use phrases like "I believe...", "As of my training cutoff...", '
    '"This is contested..." rather than presenting uncertain information '
    "with false confidence."
)

_NO_FLUFF = (
    "No preamble, no closing remarks, no offers of further help. Begin directly with ANSWER:."
)


def _word_range(fmt: ResponseFormat) -> str:
    lo, hi = _WORD_TARGETS[fmt]
    return f"{lo}-{hi} words"


# ---------------------------------------------------------------------------
# Standard model templates (used by llm plugin, and as fallback)
# ---------------------------------------------------------------------------


def _standard_system(fmt: ResponseFormat) -> str:
    return f"""\
You are an expert research analyst answering a single question with no follow-up.

{_STRUCTURE_BLOCK}

Constraints:
- Total response: {_word_range(fmt)}
- {_UNCERTAINTY_RULE}
- {_NO_FLUFF}"""


# ---------------------------------------------------------------------------
# Per-provider variants
# ---------------------------------------------------------------------------


def _perplexity_system(fmt: ResponseFormat) -> str:
    return f"""\
You are an expert research analyst with access to current web sources.

{_STRUCTURE_BLOCK}

After KEY POINTS, add:

SOURCES:
[1] URL or source title
[2] ...

Constraints:
- Response body (excluding sources): {_word_range(fmt)}
- Cite at least 3 distinct sources inline as [1], [2], etc.
- Prioritise recent sources where recency matters.
- If sources conflict, note conflicts in CAVEATS.
- {_UNCERTAINTY_RULE}
- {_NO_FLUFF}"""


def _deepseek_system(fmt: ResponseFormat) -> str:
    # DeepSeek-reasoner does internal CoT — don't ask it to show reasoning.
    return f"""\
You are an expert research analyst answering a single question with no follow-up.

{_STRUCTURE_BLOCK}

Constraints:
- Total response: {_word_range(fmt)}
- Do NOT include chain-of-thought or step-by-step reasoning in your answer — \
your internal reasoning is already captured separately.
- {_UNCERTAINTY_RULE}
- {_NO_FLUFF}"""


# ---------------------------------------------------------------------------
# Deep research templates — structured but longer
# ---------------------------------------------------------------------------

_DEEP_RESEARCH_TEMPLATE = """\
Conduct focused research on the following question and provide a structured report.

Your response MUST use exactly this structure:

SYNTHESIS (150-200 words):
The definitive answer based on your research.

EVIDENCE (5 bullet points maximum, 60 words each):
The strongest evidence supporting your synthesis, with source attribution.

COUNTERARGUMENTS (100 words maximum):
The strongest evidence against your synthesis. If none, explain why.

CONFIDENCE AND CAVEATS (100 words maximum):
What you are uncertain about, where sources conflict, or where your research is limited.

SOURCES:
Minimum 5 sources with URLs where available.

Total target: 800-1200 words. Do not pad with background context. Begin directly."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEEP_SOURCES = frozenset({"openai-deep", "google-deep"})


def get_system_prompt(source: str, fmt: ResponseFormat = ResponseFormat.STANDARD) -> str:
    """Return an appropriate system prompt for the given provider source and format.

    Parameters
    ----------
    source:
        The provider source key (e.g. "llm", "perplexity", "openai-deep").
    fmt:
        Desired response length.
    """
    if source in _DEEP_SOURCES:
        return _DEEP_RESEARCH_TEMPLATE

    if source == "perplexity":
        return _perplexity_system(fmt)

    if source == "deepseek":
        return _deepseek_system(fmt)

    # "llm", "xai", or anything else → generic structured template
    return _standard_system(fmt)
