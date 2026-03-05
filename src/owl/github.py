"""GitHub Issues integration - post council responses as comments."""

from __future__ import annotations

import os

import httpx

from .providers.base import OwlResponse

GITHUB_API = "https://api.github.com"


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


def _format_comment(response: OwlResponse) -> str:
    """Format an OwlResponse as a GitHub comment."""
    source_label = response.source
    if response.source == "llm":
        source_label = "llm plugin"

    lines = [f"## 🦉 {response.model_name}"]
    lines.append(f"*Source: {source_label}*\n")

    if response.error:
        lines.append(f"**Error:** {response.error}")
    else:
        if response.reasoning:
            lines.append("<details><summary>Reasoning</summary>\n")
            lines.append(response.reasoning)
            lines.append("\n</details>\n")

        lines.append(response.text)

        if response.citations:
            lines.append("\n### Sources")
            for citation in response.citations:
                lines.append(f"- {citation}")

    lines.append("\n---")
    lines.append("*Posted by [Parliament of Owls](https://github.com/joelio/owl)*")
    return "\n".join(lines)


async def create_issue(
    repo: str, title: str, body: str, token: str | None = None
) -> int:
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


async def post_comment(
    repo: str, issue_number: int, body: str, token: str | None = None
) -> None:
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
    """Post all council responses to a GitHub issue (new or existing)."""
    if issue_number is None:
        title = prompt[:100] + ("..." if len(prompt) > 100 else "")
        body = f"## 🦉 Parliament of Owls Query\n\n{prompt}"
        issue_number = await create_issue(repo, title, body)

    for response in responses:
        comment = _format_comment(response)
        await post_comment(repo, issue_number, comment)

    return issue_number
