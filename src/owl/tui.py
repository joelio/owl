"""Interactive TUI for selecting council members."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from .config import Config, CouncilMember, load_config, save_config
from .models import AvailableModel, discover_all_models

# With OpenRouter installed the model list runs to several hundred entries.
# Reprinting all of them on every keystroke is unusable, so the table is
# capped and the filter is how you reach anything past the cap.
MAX_VISIBLE = 30


def _key(model: AvailableModel | CouncilMember) -> str:
    return f"{model.source}:{model.name}"


@dataclass
class Selection:
    """Selection state for the picker, kept apart from the input loop.

    Separated so the behaviour can be tested without driving stdin; the
    loop is then only responsible for reading a line and printing a table.
    """

    models: list[AvailableModel]
    selected: set[str] = field(default_factory=set)
    query: str = ""

    @classmethod
    def from_config(cls, models: list[AvailableModel], config: Config) -> Selection:
        chosen = {_key(m) for m in config.council}
        return cls(models=models, selected={_key(m) for m in models if _key(m) in chosen})

    def matching(self) -> list[AvailableModel]:
        """Every model matching the current filter, uncapped."""
        if not self.query:
            return self.models
        q = self.query.lower()
        return [m for m in self.models if q in m.name.lower() or q in m.source.lower()]

    def visible(self) -> list[AvailableModel]:
        """Models matching the current filter, capped for display."""
        return self.matching()[:MAX_VISIBLE]

    def hidden_count(self) -> int:
        return max(len(self.matching()) - MAX_VISIBLE, 0)

    def is_selected(self, model: AvailableModel) -> bool:
        return _key(model) in self.selected

    def toggle(self, position: int) -> bool:
        """Toggle the model at a 1-based position in the visible list."""
        visible = self.visible()
        if not 1 <= position <= len(visible):
            return False
        key = _key(visible[position - 1])
        if key in self.selected:
            self.selected.remove(key)
        else:
            self.selected.add(key)
        return True

    def select_visible(self) -> None:
        """Select everything currently shown, not the whole catalogue.

        Filtering to ``:free`` then pressing ``a`` is the point of this.
        """
        self.selected.update(_key(m) for m in self.visible())

    def clear_visible(self) -> None:
        self.selected.difference_update(_key(m) for m in self.visible())

    def council(self) -> list[CouncilMember]:
        """Selected models, in catalogue order, regardless of the filter."""
        return [
            CouncilMember(name=m.name, source=m.source) for m in self.models if self.is_selected(m)
        ]


def _render(console: Console, selection: Selection) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=4)
    table.add_column("", width=3)
    table.add_column("Model", min_width=25)
    table.add_column("Source", min_width=12)
    table.add_column("Description")

    last_category = None
    for index, model in enumerate(selection.visible(), start=1):
        if model.category != last_category:
            label = "Standard Models (via llm)" if model.category == "standard" else "Deep Research"
            colour = "cyan" if model.category == "standard" else "magenta"
            table.add_row("", "", f"[bold {colour}]{label}[/bold {colour}]", "", "")
            last_category = model.category
        check = "☑" if selection.is_selected(model) else "☐"
        table.add_row(str(index), check, model.name, model.source, model.description)

    console.print(table)

    hidden = selection.hidden_count()
    if hidden:
        console.print(f"[dim]...and {hidden} more. Narrow with /text to reach them.[/dim]")
    if selection.query:
        console.print(f"[dim]Filter: [bold]{selection.query}[/bold] — / alone clears it[/dim]")
    console.print(f"[dim]{len(selection.selected)} selected[/dim]")


def run_council_selector() -> Config:
    """Interactive council member selector using rich."""
    console = Console()
    config = load_config()
    available = discover_all_models()

    if not available:
        console.print("[red]No models found![/red] Install llm plugins or set API keys.")
        return config

    # Standard models first, then deep research, so numbering is predictable.
    ordered = [m for m in available if m.category == "standard"] + [
        m for m in available if m.category != "standard"
    ]
    selection = Selection.from_config(ordered, config)

    console.print()
    console.print("[bold]🦉 Parliament of Owls — Select Your Council[/bold]")
    console.print()

    while True:
        _render(console, selection)
        console.print()
        console.print(
            "Number to toggle, [bold]/text[/bold]=filter, [bold]a[/bold]=all shown, "
            "[bold]n[/bold]=none shown, [bold]s[/bold]=save, [bold]q[/bold]=quit: ",
            end="",
        )

        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D or Ctrl-C should leave the config alone, not traceback.
            console.print("\n[yellow]Cancelled.[/yellow] Council unchanged.")
            return config

        lowered = choice.lower()
        if lowered == "q":
            console.print("[yellow]Cancelled.[/yellow] Council unchanged.")
            return config
        if lowered == "s":
            break
        if choice.startswith("/"):
            selection.query = choice[1:].strip()
        elif lowered == "a":
            selection.select_visible()
        elif lowered == "n":
            selection.clear_visible()
        elif choice.isdigit() and not selection.toggle(int(choice)):
            console.print("[red]No such row.[/red]")

    config.council = selection.council()
    save_config(config)
    console.print(f"\n[green]✓ Council saved with {len(config.council)} members.[/green]")
    return config
