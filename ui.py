# ui.py
"""Terminal UI — all rich rendering, input handling, spinners."""

from __future__ import annotations

import readline

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console()


def print_banner(model: str, server: str, context_length: int) -> None:
    """Display the startup banner with model info."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Model", model)
    table.add_row("Server", server)
    table.add_row("Context", f"{context_length:,} tokens")

    panel = Panel(
        table,
        title="[bold]llama-chat[/bold]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_model_menu(servers: list[dict]) -> None:
    """Display numbered model selection menu.

    Each dict has: number, label, server, status.
    """
    console.print("\n[bold]Select a model:[/bold]")
    for s in servers:
        number = s["number"]
        label = s["label"]
        server = s["server"]
        status = s["status"]

        if status == "online":
            console.print(f"  [bold][{number}][/bold] {label} ({server})")
        elif status == "loading":
            console.print(
                f"  [bold][{number}][/bold] {label} ({server}) [yellow](loading...)[/yellow]"
            )
        else:
            console.print(f"  [dim][ ] {label} ({server}) (offline)[/dim]")


def get_model_choice(max_num: int, selectable: set[int]) -> int:
    """Get the user's model selection. Returns 1-indexed choice."""
    while True:
        try:
            raw = input("\nChoice: ").strip()
            choice = int(raw)
            if choice in selectable:
                return choice
            console.print("[red]That model is not available.[/red]")
        except (ValueError, EOFError):
            console.print(f"[red]Enter a number from the menu.[/red]")


def get_input(prompt: str = "You > ", prefill: str = "") -> str:
    """Get user input with optional pre-fill via readline."""
    if prefill:
        readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def confirm(message: str) -> bool:
    """Ask a yes/no question. Returns True for yes."""
    response = input(f"{message} (y/n) ").strip().lower()
    return response in ("y", "yes")


class SpinnerDisplay:
    """Manages a spinner with queue status and elapsed time."""

    def __init__(self) -> None:
        self._live: Live | None = None

    def start(self) -> None:
        """Show 'Thinking...' spinner."""
        self._live = Live(
            Spinner("dots", text="Thinking..."),
            console=console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()

    def set_queued(self, server: str, elapsed: float) -> None:
        """Update spinner to show queue status with elapsed time."""
        if self._live:
            text = f"Queued — waiting for an available slot on {server} ({elapsed:.0f}s)..."
            self._live.update(Spinner("dots", text=text))

    def stop(self) -> None:
        """Stop the spinner."""
        if self._live:
            self._live.stop()
            self._live = None


class StreamingDisplay:
    """Manages rich.live.Live for streaming markdown rendering."""

    def __init__(self) -> None:
        self._content: str = ""
        self._live: Live | None = None

    def start(self) -> None:
        """Begin streaming display."""
        self._content = ""
        self._live = Live(
            Text(""),
            console=console,
            refresh_per_second=10,
        )
        self._live.start()

    def update(self, token: str) -> None:
        """Append a token and re-render."""
        self._content += token
        if self._live:
            self._live.update(Markdown(self._content))

    def stop(self) -> None:
        """Final render and stop."""
        if self._live:
            self._live.update(Markdown(self._content))
            self._live.stop()
            self._live = None


def print_stats(token_count: int, duration: float) -> None:
    """Display token/s statistics after a response."""
    if duration > 0:
        speed = token_count / duration
        console.print(
            f"[dim]{token_count} tokens in {duration:.1f}s ({speed:.1f} tok/s)[/dim]"
        )
    elif token_count > 0:
        console.print(f"[dim]{token_count} tokens[/dim]")


def print_error(message: str) -> None:
    """Display an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_muted(message: str) -> None:
    """Display a dim/muted message."""
    console.print(f"[dim]{message}[/dim]")


def print_help(config_path: str) -> None:
    """Display available commands and config path."""
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Command", style="bold")
    table.add_column("Action")
    table.add_row("/help", "Show this help message")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/system <prompt>", "Change the system prompt")
    table.add_row("/model", "Switch to a different model")
    table.add_row("/quit", "Save and exit")

    console.print()
    console.print(table)
    console.print(f"\n[dim]Config: {config_path}[/dim]")
    console.print()
