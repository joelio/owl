"""Reconcile the council's answers into one, using an arbiter model.

Fanning out to several models and printing every answer leaves the reader to
do the reconciling.  This module hands that job to one more model, which is
given the council's answers in full rather than a tally.

Reading the answers matters more than counting them: a council can agree and
still be wrong, and the reasoning in a dissenting answer is often what shows
it.  So the arbiter is asked to weigh reasoning, name genuine disagreement,
and flag claims that only one member made.
"""

from __future__ import annotations

import logging

from .config import CouncilMember
from .prompts import get_arbiter_prompt
from .providers.base import OwlResponse
from .providers.errors import describe_error
from .providers.registry import get_provider

logger = logging.getLogger(__name__)

SYNTHESIS_SOURCE = "synthesis"

# Keep the arbiter's own input bounded.  A full council of deep research
# reports can run to tens of thousands of words, and the arbiter has a
# context limit like anything else.
_MAX_CHARS_PER_ANSWER = 12_000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CHARS_PER_ANSWER:
        return text
    return text[:_MAX_CHARS_PER_ANSWER] + "\n\n[...answer truncated for the arbiter]"


def build_arbiter_input(prompt: str, responses: list[OwlResponse]) -> str:
    """Lay out the original question and every successful answer."""
    blocks = [
        "ORIGINAL QUESTION",
        "",
        prompt.strip(),
        "",
        "COUNCIL ANSWERS",
        "",
    ]
    for index, response in enumerate(responses, start=1):
        blocks.append(f"--- Member {index}: {response.model_name} ({response.source}) ---")
        blocks.append("")
        if response.reasoning:
            blocks.append("Its stated reasoning:")
            blocks.append(_truncate(response.reasoning))
            blocks.append("")
        blocks.append(_truncate(response.text))
        blocks.append("")
    return "\n".join(blocks)


async def synthesise(
    prompt: str,
    responses: list[OwlResponse],
    arbiter: CouncilMember,
) -> OwlResponse | None:
    """Ask the arbiter to reconcile the council's answers.

    Returns ``None`` when there is nothing to reconcile — no successful
    answers, or only one, in which case a synthesis pass would cost a
    request to restate what the reader already has.
    """
    answered = [r for r in responses if not r.error and r.text.strip()]
    if len(answered) < 2:
        logger.info("Skipping synthesis: %d usable answer(s)", len(answered))
        return None

    provider = get_provider(arbiter)
    arbiter_input = build_arbiter_input(prompt, answered)

    try:
        result = await provider.query(arbiter_input, system_prompt=get_arbiter_prompt())
    except Exception as e:
        logger.exception("Arbiter %s failed", arbiter.name)
        return OwlResponse(
            model_name=arbiter.name,
            source=SYNTHESIS_SOURCE,
            text="",
            error=describe_error(e),
        )

    # Present the result as a synthesis, keeping the arbiter's identity so
    # the reader can see which model reconciled the answers.
    return OwlResponse(
        model_name=arbiter.name,
        source=SYNTHESIS_SOURCE,
        text=result.text,
        error=result.error,
        citations=result.citations,
        elapsed_seconds=result.elapsed_seconds,
    )
