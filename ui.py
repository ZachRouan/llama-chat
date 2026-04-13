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
        self._reasoning: str = ""
        self._content: str = ""
        self._live: Live | None = None
        self._in_reasoning: bool = True

    def start(self) -> None:
        """Begin streaming display."""
        self._reasoning = ""
        self._content = ""
        self._in_reasoning = True
        self._live = Live(
            Text(""),
            console=console,
            refresh_per_second=10,
        )
        self._live.start()

    def update(self, token: str, is_reasoning: bool = False) -> None:
        """Append a token and re-render."""
        if is_reasoning:
            self._reasoning += token
        else:
            if self._in_reasoning and self._reasoning:
                # Transition from reasoning to content
                self._in_reasoning = False
            self._content += token

        if self._live:
            self._live.update(self._render())

    def _render(self) -> Text | Markdown:
        """Render current state with reasoning dimmed."""
        if self._in_reasoning:
            # Still in reasoning phase - show dim italic text
            text = Text(self._reasoning, style="dim italic")
            return text
        elif self._reasoning:
            # Have both reasoning and content - show reasoning dimmed, then content
            result = Text()
            result.append(self._reasoning.rstrip() + "\n\n", style="dim italic")
            # For content, render as markdown by returning combined
            # Actually, rich doesn't let us easily combine Text and Markdown
            # So we'll just show reasoning as dim prefix, then markdown content
            return result + Text.from_markup(f"[/dim][/italic]") if not self._content else Markdown(self._content)
        else:
            # No reasoning, just content
            return Markdown(self._content)

    def stop(self) -> None:
        """Final render and stop."""
        if self._live:
            # Final render: show reasoning collapsed/dimmed, then full markdown content
            if self._reasoning and self._content:
                # Print reasoning as dim block, then content as markdown
                self._live.update(Text(""))  # Clear live display
                self._live.stop()
                console.print(Text(self._reasoning.rstrip(), style="dim italic"))
                console.print()
                console.print(Markdown(self._content))
            elif self._content:
                self._live.update(Markdown(self._content))
                self._live.stop()
            else:
                self._live.stop()
            self._live = None


def print_stats(
    token_count: int,
    duration: float,
    truncated: bool = False,
    context_used: int | None = None,
    context_max: int | None = None,
) -> None:
    """Display token/s statistics and context usage after a response."""
    parts = []
    if duration > 0:
        speed = token_count / duration
        parts.append(f"{token_count} tokens in {duration:.1f}s ({speed:.1f} tok/s)")
    elif token_count > 0:
        parts.append(f"{token_count} tokens")

    if context_used is not None and context_max is not None:
        pct = (context_used / context_max) * 100
        parts.append(f"context: {context_used:,}/{context_max:,} ({pct:.0f}%)")

    if parts:
        console.print(f"[dim]{' · '.join(parts)}[/dim]")

    if truncated:
        console.print(
            "[yellow](Response truncated — hit max_tokens limit. "
            "Increase LLAMA_MAX_TOKENS in config to allow longer responses.)[/yellow]"
        )


def print_error(message: str) -> None:
    """Display an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_muted(message: str) -> None:
    """Display a dim/muted message."""
    console.print(f"[dim]{message}[/dim]")


def print_tool_call(name: str, arguments: dict) -> None:
    """Display a tool call in dim style."""
    if name == "read_file":
        summary = arguments.get("path", "")
    elif name == "write_file":
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        lines = content.count("\n") + 1
        summary = f"{path} ({lines} lines)"
    elif name == "run_command":
        summary = arguments.get("command", "")
    elif name == "list_directory":
        summary = arguments.get("path", ".")
    elif name == "search_files":
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        summary = f'"{pattern}" in {path}'
    else:
        summary = str(arguments)[:100]
    console.print(f"[dim]→ {name}: {summary}[/dim]")


def print_tool_result(name: str, result: str, max_lines: int = 5) -> None:
    """Display a brief tool result in dim style."""
    if result.startswith("Error:"):
        console.print(f"[red]  \u2717 {result}[/red]")
    else:
        result_lines = result.rstrip("\n").split("\n")
        total = len(result_lines)
        if total <= max_lines:
            for line in result_lines:
                console.print(f"[dim]  \u2713 {line}[/dim]", highlight=False)
        else:
            for line in result_lines[:max_lines]:
                console.print(f"[dim]  \u2713 {line}[/dim]", highlight=False)
            remaining = total - max_lines
            console.print(f"[dim]  \u2713 ... ({remaining} more lines)[/dim]")


def print_command_output(line: str) -> None:
    """Display a single line of command output in dim style."""
    console.print(f"[dim]  {line.rstrip()}[/dim]", highlight=False)


def prompt_tool_permission(name: str, summary: str) -> str:
    """Prompt user to allow a tool call. Returns 'yes', 'no', or 'always'."""
    console.print(f"[yellow]  ⚠ {name}: {summary}[/yellow]")
    while True:
        response = input("  Allow? (y)es / (n)o / (a)lways: ").strip().lower()
        if response in ("y", "yes"):
            return "yes"
        elif response in ("n", "no"):
            return "no"
        elif response in ("a", "always"):
            return "always"
        console.print("[dim]  Enter y, n, or a.[/dim]")


def print_broad_pattern_warning(tool_name: str) -> None:
    """Warn the user when 'always' would create a very broad rule."""
    if tool_name == "write_file":
        console.print("[yellow]  ⚠ This will allow all writes in the current directory.[/yellow]")


def print_permissions(permissions: dict) -> None:
    """Display current permission defaults and allow rules."""
    console.print()
    console.print("[bold]Tool defaults:[/bold]")
    for tool in ["read_file", "list_directory", "search_files", "write_file", "run_command"]:
        value = permissions.get(tool, "ask")
        style = "green" if value == "allow" else "yellow"
        console.print(f"  {tool:<16} [{style}]{value}[/{style}]")

    rules = permissions.get("allow_rules", [])
    if rules:
        console.print()
        console.print("[bold]Allow rules:[/bold]")
        for i, rule in enumerate(rules):
            tool = rule.get("tool", "?")
            pattern = rule.get("pattern", "?")
            console.print(f"  {i + 1}. {tool}: {pattern}")
    else:
        console.print()
        console.print("[dim]No allow rules configured.[/dim]")
    console.print()


def print_help(config_path: str) -> None:
    """Display available commands and config path."""
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Command", style="bold")
    table.add_column("Action")
    table.add_row("/help", "Show this help message")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/system <prompt>", "Change the system prompt")
    table.add_row("/model", "Switch to a different model")
    table.add_row("/agent", "Toggle coding agent mode (tool use)")
    table.add_row("/permissions", "Show or manage tool permission rules")
    table.add_row("/quit", "Save and exit")

    console.print()
    console.print(table)
    console.print(f"\n[dim]Config: {config_path}[/dim]")
    console.print()
