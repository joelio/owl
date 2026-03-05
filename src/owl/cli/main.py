"""Parliament of Owls CLI."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from .. import __version__
from ..config import load_config
from ..council import convene
from ..github import post_responses_to_github
from ..models import discover_all_models
from ..output import print_responses
from ..tui import run_council_selector

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="owl")
def cli() -> None:
    """🦉 Parliament of Owls - Query multiple LLMs in parallel."""
    pass


@cli.command()
@click.argument("prompt", required=False)
@click.option(
    "-f",
    "--file",
    "file_path",
    default=None,
    type=click.Path(exists=True),
    help="Read prompt from file",
)
@click.option("--gh", "github_repo", default=None, help="Post to GitHub repo (owner/repo)")
@click.option("--issue", "issue_number", default=None, type=int, help="Existing issue number")
def ask(
    prompt: str | None, file_path: str | None, github_repo: str | None, issue_number: int | None
) -> None:
    """Ask the council a question. Reads from --file, stdin pipe, or argument."""
    if file_path:
        with open(file_path) as f:
            prompt = f.read().strip()
    elif prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            console.print(
                "[red]No prompt provided.[/red] Pass as argument, --file, or pipe via stdin."
            )
            return

    if not prompt:
        console.print("[red]Empty prompt.[/red]")
        return

    config = load_config()

    if not config.council:
        console.print(
            "[yellow]No council members configured.[/yellow] Run [bold]owl council[/bold] first."
        )
        return

    console.print(f"[dim]Querying {len(config.council)} council members...[/dim]")

    responses = asyncio.run(convene(prompt, config))
    print_responses(responses, console)

    if github_repo:
        console.print(f"\n[dim]Posting to {github_repo}...[/dim]")
        result_issue = asyncio.run(
            post_responses_to_github(responses, github_repo, issue_number, prompt)
        )
        console.print(
            f"[green]✓ Posted to https://github.com/{github_repo}/issues/{result_issue}[/green]"
        )


@cli.command()
def council() -> None:
    """Select which models are in your council."""
    run_council_selector()


@cli.command(name="council-list")
def council_list() -> None:
    """Show current council members."""
    config = load_config()

    if not config.council:
        console.print(
            "[yellow]No council members configured.[/yellow] Run [bold]owl council[/bold] first."
        )
        return

    console.print("\n[bold]🦉 Current Council[/bold]\n")
    for member in config.council:
        console.print(f"  • {member.name} [dim]({member.source})[/dim]")
    console.print()


@cli.command()
def models() -> None:
    """Show all available models."""
    available = discover_all_models()

    if not available:
        console.print("[yellow]No models found.[/yellow] Install llm plugins or set API keys.")
        return

    standard = [m for m in available if m.category == "standard"]
    deep = [m for m in available if m.category == "deep-research"]

    console.print("\n[bold]🦉 Available Models[/bold]\n")

    if standard:
        console.print("[bold cyan]Standard (via llm)[/bold cyan]")
        for m in standard:
            console.print(f"  • {m.name} [dim]{m.description}[/dim]")

    if deep:
        console.print("\n[bold magenta]Deep Research[/bold magenta]")
        for m in deep:
            console.print(f"  • {m.name} [dim]{m.description}[/dim]")

    console.print()
