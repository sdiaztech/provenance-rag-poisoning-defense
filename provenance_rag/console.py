"""Terminal output with rich."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .data_sources import DataSource


class ConsoleReport:
    """Prints data sources as tables, and keeps a copy of everything printed so it can be saved as a log."""

    def __init__(self, max_chars: int = 120) -> None:
        self.console = Console(record=True)
        self.max_chars = max_chars

    def save_log(self, logs_dir: Path) -> Path:
        """Saves everything printed this run as plain text: a timestamped file plus latest.txt."""
        logs_dir.mkdir(parents=True, exist_ok=True)
        text = self.console.export_text(clear=False)
        path = logs_dir / f"run-{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        for target in (path, logs_dir / "latest.txt"):
            target.write_text(text, encoding="utf-8")
        return path

    def show_source(self, source: DataSource, rows: int = 5) -> None:
        table = source.table
        self.console.rule(Text(source.title, style="bold"))
        self.console.print(f"{len(table):,} rows x {table.shape[1]} columns. First {rows} rows:")
        self.console.print(self._as_table(source.sample(rows)))
        self.console.print(Text("All columns: " + ", ".join(map(str, table.columns)), style="dim"))
        self.console.print()

    def show_table(self, title: str, frame: pd.DataFrame, note: str = "") -> None:
        self.console.rule(Text(title, style="bold"))
        if note:
            self.console.print(Text(note))
        self.console.print(self._as_table(frame))
        self.console.print()

    def show_note(self, text: str, style: str = "yellow") -> None:
        self.console.print(Text(text, style=style))
        self.console.print()

    def _as_table(self, frame: pd.DataFrame) -> Table:
        table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=True)
        cells = frame.map(self._shorten)
        for column in cells.columns:
            single_word = not cells[column].str.contains(" ").any()  # IDs and dates stay on one line
            table.add_column(str(column), no_wrap=single_word,
                             min_width=int(cells[column].str.len().max()) if single_word else None)
        for _, row in cells.iterrows():
            table.add_row(*(Text(value) for value in row))
        return table

    def _shorten(self, value: object) -> str:
        text = "" if pd.isna(value) else " ".join(str(value).split())
        return text if len(text) <= self.max_chars else text[: self.max_chars - 1] + "…"
