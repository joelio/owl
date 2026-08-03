"""GitHub Issues integration - post council responses as a single consolidated comment."""

from __future__ import annotations

import os
from dataclasses import replace

import httpx

from .providers.base import OwlResponse

GITHUB_API = "https://api.github.com"

# GitHub comment body limit is 65536 chars; leave headroom
_MAX_COMMENT_CHARS = 62_000

_TRUNCATION_NOTE = "\n\n_[truncated: response exceeded GitHub's comment size limit]_"


def _get_token() -> str:
    """Get GitHub token from env or gh CLI."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        # Try gh CLI auth
        import subprocess

        try:
            result = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, check=True
            )
            token = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return token


def _format_response_section(response: OwlResponse) -> str:
    """Format a single response as a collapsible <details> section."""
    source_label = "llm plugin" if response.source == "llm" else response.source
    timing = f" — {response.elapsed_seconds}s" if response.elapsed_seconds else ""

    summary = (
        f"<summary><strong>🦉 {response.model_name}</strong> "
        f"<em>({source_label}{timing})</em></summary>"
    )

    lines = [
        "<details>",
        summary,
        "",
    ]

    if response.reasoning:
        lines.append("<details>")
        lines.append("<summary>Reasoning</summary>")
        lines.append("")
        lines.append(response.reasoning)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append(response.text)

    if response.citations:
        lines.append("")
        lines.append("#### Sources")
        for citation in response.citations:
            lines.append(f"- {citation}")

    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def _fit_section(response: OwlResponse, budget: int) -> str:
    """Render a response as a section that fits within ``budget`` characters.

    Splitting between sections is not enough on its own: one response longer
    than a whole comment can never be placed, and GitHub rejects the
    oversized body with a 422 that loses every other response with it.

    Content is shed in order of how much it costs the reader: reasoning is
    supplementary, citations are recoverable, the answer itself is trimmed
    last.  Each step re-renders rather than slicing the markup, so the
    ``<details>`` tags stay balanced.
    """
    section = _format_response_section(response)
    if len(section) <= budget:
        return section

    if response.reasoning:
        response = replace(response, reasoning=None)
        section = _format_response_section(response)
        if len(section) <= budget:
            return section

    if response.citations:
        response = replace(response, citations=None)
        section = _format_response_section(response)
        if len(section) <= budget:
            return section

    overhead = len(section) - len(response.text)
    keep = max(budget - overhead - len(_TRUNCATION_NOTE), 0)
    section = _format_response_section(
        replace(response, text=response.text[:keep] + _TRUNCATION_NOTE)
    )

    # Pathological case: the wrapper alone does not fit. Nothing renders
    # usefully at this size, but the body must still be postable.
    return section if len(section) <= budget else section[:budget]


def _build_consolidated_comment(responses: list[OwlResponse]) -> list[str]:
    """Build one or more comment bodies with all responses collapsed.

    Returns a list of comment bodies.  Usually one, but splits into
    multiple if the combined text would exceed GitHub's limit.
    """
    success = [r for r in responses if not r.error]
    errors = [r for r in responses if r.error]

    header_lines = [
        "## 🦉 Council Response",
        "",
        f"**{len(success)} of {len(responses)} members responded**",
        "",
        "---",
        "",
    ]
    header = "\n".join(header_lines)

    footer_lines = ["", "---"]
    if errors:
        footer_lines.append("")
        footer_lines.append("### ⚠️ Errors")
        for r in errors:
            footer_lines.append(f"- **{r.model_name}** ({r.source}): {r.error}")
    footer_lines.append("")
    footer_lines.append("*Posted by [Parliament of Owls](https://github.com/joelio/owl)*")
    footer = "\n".join(footer_lines)

    # Every section must be placeable in a comment of its own, so cap each
    # one at what is left after the header and footer are accounted for.
    section_budget = _MAX_COMMENT_CHARS - len(header) - len(footer)
    sections = [_fit_section(r, section_budget) for r in success]

    # Try to fit everything in one comment
    combined = header + "\n".join(sections) + footer
    if len(combined) <= _MAX_COMMENT_CHARS:
        return [combined]

    # Split into multiple comments if too large
    comments: list[str] = []
    current = header
    for section in sections:
        if len(current) + len(section) + len(footer) > _MAX_COMMENT_CHARS and current != header:
            comments.append(current + footer)
            current = header
        current += section + "\n"
    current += footer
    comments.append(current)
    return comments


async def create_issue(repo: str, title: str, body: str, token: str | None = None) -> int:
    """Create a GitHub issue and return its number."""
    token = token or _get_token()
    if not token:
        raise RuntimeError("No GitHub token available. Set GITHUB_TOKEN or install gh CLI.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{GITHUB_API}/repos/{repo}/issues",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json={"title": title, "body": body},
        )
        resp.raise_for_status()
        return resp.json()["number"]


async def post_comment(repo: str, issue_number: int, body: str, token: str | None = None) -> None:
    """Post a comment to a GitHub issue."""
    token = token or _get_token()
    if not token:
        raise RuntimeError("No GitHub token available. Set GITHUB_TOKEN or install gh CLI.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{GITHUB_API}/repos/{repo}/issues/{issue_number}/comments",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json={"body": body},
        )
        resp.raise_for_status()


async def post_responses_to_github(
    responses: list[OwlResponse],
    repo: str,
    issue_number: int | None = None,
    prompt: str = "",
) -> int:
    """Post all council responses to a GitHub issue as collapsed sections."""
    if issue_number is None:
        title = prompt[:100] + ("..." if len(prompt) > 100 else "")
        body = f"## 🦉 Parliament of Owls Query\n\n{prompt}"
        issue_number = await create_issue(repo, title, body)

    comment_bodies = _build_consolidated_comment(responses)
    for comment_body in comment_bodies:
        await post_comment(repo, issue_number, comment_body)

    return issue_number
