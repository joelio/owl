"""Interactive TUI for selecting council members."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from .config import Config, CouncilMember, load_config, save_config
from .models import AvailableModel, discover_all_models


def _model_is_selected(model: AvailableModel, config: Config) -> bool:
    return any(m.name == model.name and m.source == model.source for m in config.council)


def run_council_selector() -> Config:
    """Interactive council member selector using rich."""
    console = Console()
    config = load_config()
    available = discover_all_models()

    if not available:
        console.print("[red]No models found![/red] Install llm plugins or set API keys.")
        return config

    # Group by category
    standard = [m for m in available if m.category == "standard"]
    deep = [m for m in available if m.category == "deep-research"]

    # Build selection state
    selected: dict[str, bool] = {}
    for model in available:
        key = f"{model.source}:{model.name}"
        selected[key] = _model_is_selected(model, config)

    console.print()
    console.print("[bold]🦉 Parliament of Owls — Select Your Council[/bold]")
    console.print()

    def show_table() -> None:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=4)
        table.add_column("", width=3)
        table.add_column("Model", min_width=25)
        table.add_column("Source", min_width=12)
        table.add_column("Description")

        idx = 1
        if standard:
            table.add_row("", "", "[bold cyan]Standard Models (via llm)[/bold cyan]", "", "")
            for model in standard:
                key = f"{model.source}:{model.name}"
                check = "☑" if selected[key] else "☐"
                table.add_row(str(idx), check, model.name, model.source, model.description)
                idx += 1

        if deep:
            table.add_row("", "", "[bold magenta]Deep Research[/bold magenta]", "", "")
            for model in deep:
                key = f"{model.source}:{model.name}"
                check = "☑" if selected[key] else "☐"
                table.add_row(str(idx), check, model.name, model.source, model.description)
                idx += 1

        console.print(table)

    all_models = standard + deep

    while True:
        show_table()
        console.print()
        console.print("Enter number to toggle, [bold]a[/bold]=all, [bold]n[/bold]=none, "
                       "[bold]s[/bold]=save, [bold]q[/bold]=quit: ", end="")

        choice = input().strip().lower()

        if choice == "q":
            return config
        elif choice == "s":
            break
        elif choice == "a":
            for model in all_models:
                selected[f"{model.source}:{model.name}"] = True
        elif choice == "n":
            for model in all_models:
                selected[f"{model.source}:{model.name}"] = False
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_models):
                model = all_models[idx]
                key = f"{model.source}:{model.name}"
                selected[key] = not selected[key]

    # Build new config from selections
    new_council = []
    for model in all_models:
        key = f"{model.source}:{model.name}"
        if selected[key]:
            new_council.append(CouncilMember(name=model.name, source=model.source))

    config.council = new_council
    save_config(config)
    console.print(f"\n[green]✓ Council saved with {len(new_council)} members.[/green]")
    return config
