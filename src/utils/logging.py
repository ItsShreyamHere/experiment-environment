"""Console logging via rich, with a plain fallback.

Mirrors the family convention so output looks consistent across siblings.
"""

from __future__ import annotations

try:
    from rich.console import Console

    _console = Console()

    def info(msg: str) -> None:
        _console.print(msg)

    def warn(msg: str) -> None:
        _console.print(f"[yellow]{msg}[/yellow]")

    def error(msg: str) -> None:
        _console.print(f"[red]{msg}[/red]")

    def success(msg: str) -> None:
        _console.print(f"[green]{msg}[/green]")

    def dim(msg: str) -> None:
        _console.print(f"[dim]{msg}[/dim]")

except Exception:  # pragma: no cover - rich should always be present
    def info(msg: str) -> None:
        print(msg)

    def warn(msg: str) -> None:
        print(f"WARN: {msg}")

    def error(msg: str) -> None:
        print(f"ERROR: {msg}")

    def success(msg: str) -> None:
        print(msg)

    def dim(msg: str) -> None:
        print(msg)
