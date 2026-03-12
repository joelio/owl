"""GitHub Issues integration - post council responses as a single consolidated comment."""

from __future__ import annotations

import os

import httpx

from .providers.base import OwlResponse

GITHUB_API = "https://api.github.com"

# GitHub comment body limit is 65536 chars; leave headroom
_MAX_COMMENT_CHARS = 62_000


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

    lines = [
        "<details>",
        f"<summary><strong>🦉 {response.model_name}</strong> <em>({source_label}{timing})</em></summary>",
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


def _build_consolidated_comment(
    responses: list[OwlResponse],
    prompt: str,
) -> list[str]:
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

    # Build response sections
    sections = [_format_response_section(r) for r in success]

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

    async with httpx.AsyncClient() as client:
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

    async with httpx.AsyncClient() as client:
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

    comment_bodies = _build_consolidated_comment(responses, prompt)
    for comment_body in comment_bodies:
        await post_comment(repo, issue_number, comment_body)

    return issue_number
