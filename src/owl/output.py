"""Terminal output formatting using rich."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .providers.base import OwlResponse


def _timing_badge(resp: OwlResponse) -> str:
    if resp.elapsed_seconds is not None:
        return f" [dim]({resp.elapsed_seconds}s)[/dim]"
    return ""


def print_responses(responses: list[OwlResponse], console: Console | None = None) -> None:
    """Print council responses to terminal with rich formatting."""
    console = console or Console()

    success = [r for r in responses if not r.error]
    errors = [r for r in responses if r.error]

    console.print()
    console.print(
        f"[bold]🦉 Parliament of Owls — {len(success)} of {len(responses)} members responded[/bold]"
    )
    console.print()

    for response in success:
        source_tag = f"[dim]{response.source}[/dim]"
        timing = _timing_badge(response)
        title = f"🦉 {response.model_name} {source_tag}{timing}"

        content_parts = []
        if response.reasoning:
            content_parts.append("[dim]Reasoning:[/dim]\n" + response.reasoning + "\n")
        content_parts.append(response.text)
        if response.citations:
            content_parts.append("\n[bold]Sources:[/bold]")
            for cite in response.citations:
                content_parts.append(f"  • {cite}")

        panel = Panel(
            Markdown("\n".join(content_parts)),
            title=title,
            title_align="left",
            border_style="cyan",
            expand=True,
        )
        console.print(panel)

    for response in errors:
        panel = Panel(
            f"[red]{response.error}[/red]",
            title=f"🦉 {response.model_name} [red]ERROR[/red]",
            title_align="left",
            border_style="red",
            expand=True,
        )
        console.print(panel)
