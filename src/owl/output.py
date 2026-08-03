"""Terminal output formatting using rich."""

from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from .providers.base import OwlResponse


def _timing_badge(resp: OwlResponse) -> str:
    if resp.elapsed_seconds is not None:
        return f" [dim]({resp.elapsed_seconds}s)[/dim]"
    return ""


def _panel_title(response: OwlResponse) -> str:
    """Build the panel title.

    Model names come from config and from API responses, so they are escaped
    before going into a string rich will parse for markup.
    """
    source_tag = f"[dim]{escape(response.source)}[/dim]"
    return f"🦉 {escape(response.model_name)} {source_tag}{_timing_badge(response)}"


def _response_body(response: OwlResponse) -> RenderableType:
    """Assemble the panel body out of separate renderables.

    Labels are ``Text`` rather than console markup embedded in the markdown
    string: ``Markdown`` does not parse console markup, so a
    ``[dim]Reasoning:[/dim]`` prefix reached the user verbatim, tags and all.
    """
    parts: list[RenderableType] = []

    if response.reasoning:
        parts.append(Text("Reasoning", style="dim bold"))
        parts.append(Markdown(response.reasoning))
        parts.append(Text())

    parts.append(Markdown(response.text))

    if response.citations:
        parts.append(Text())
        parts.append(Text("Sources", style="bold"))
        for cite in response.citations:
            parts.append(Text(f"  • {cite}"))

    return Group(*parts)


def print_responses(
    responses: list[OwlResponse],
    console: Console | None = None,
    synthesis: OwlResponse | None = None,
) -> None:
    """Print council responses to terminal with rich formatting."""
    console = console or Console()

    success = [r for r in responses if not r.error]
    errors = [r for r in responses if r.error]

    console.print()
    console.print(
        f"[bold]🦉 Parliament of Owls — {len(success)} of {len(responses)} members responded[/bold]"
    )
    console.print()

    # The synthesis leads: it is the answer the reader wants, and the
    # individual responses below are the working behind it.
    if synthesis is not None and not synthesis.error:
        console.print(
            Panel(
                Markdown(synthesis.text),
                title=f"⚖️  Synthesis [dim]{escape(synthesis.model_name)}[/dim]"
                f"{_timing_badge(synthesis)}",
                title_align="left",
                border_style="green",
                expand=True,
            )
        )
        console.print()
    elif synthesis is not None and synthesis.error:
        console.print(
            Panel(
                Text(synthesis.error, style="red"),
                title=f"⚖️  Synthesis [red]FAILED[/red] [dim]{escape(synthesis.model_name)}[/dim]",
                title_align="left",
                border_style="red",
                expand=True,
            )
        )
        console.print()

    for response in success:
        panel = Panel(
            _response_body(response),
            title=_panel_title(response),
            title_align="left",
            border_style="cyan",
            expand=True,
        )
        console.print(panel)

    for response in errors:
        # Error text carries API payload snippets, and those contain brackets.
        # Rendering it as Text stops rich swallowing "[...]" as a style tag.
        panel = Panel(
            Text(response.error or "", style="red"),
            title=f"🦉 {escape(response.model_name)} [red]ERROR[/red]",
            title_align="left",
            border_style="red",
            expand=True,
        )
        console.print(panel)
