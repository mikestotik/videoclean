from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


STAGES: list[tuple[str, str, float]] = [
    ("validate", "Проверка файла", 0.03),
    ("normalize", "Разбор на кадры", 0.12),
    ("parse", "Разбор prompt", 0.06),
    ("detect", "Поиск объектов", 0.28),
    ("inpaint", "Удаление / inpaint", 0.28),
    ("verify", "Проверка", 0.16),
    ("encode", "Промежуточный файл", 0.10),
    ("package", "Выходные форматы", 0.05),
    ("report", "Отчёт", 0.02),
]


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "—"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class StageState:
    key: str
    title: str
    weight: float
    status: str = "pending"  # pending | run | done | error
    detail: str = ""
    seconds: float = 0.0
    current: int = 0
    total: int = 0
    started_at: float | None = None


@dataclass
class JobProgress:
    job_id: str
    headline: str
    estimated_seconds: float
    stages: dict[str, StageState] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    _live: Live | None = field(default=None, repr=False)
    _bar: Progress | None = field(default=None, repr=False)
    _bar_task: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.stages = {
            key: StageState(key=key, title=title, weight=weight) for key, title, weight in STAGES
        }
        self._bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("{task.percentage:>5.1f}%"),
            transient=False,
        )
        self._bar_task = self._bar.add_task("job", total=1.0, completed=0.0)

    def __enter__(self) -> "JobProgress":
        self._live = Live(get_renderable=self.render, refresh_per_second=8, transient=False)
        self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type:
            running = next((s for s in self.stages.values() if s.status == "run"), None)
            if running:
                running.status = "error"
                running.detail = str(exc)[:120] if exc else "failed"
        self.finished_at = time.monotonic()
        if self._live:
            self._live.update(self.render())
            self._live.stop()

    def start(self, key: str, total: int = 0, detail: str = "") -> None:
        stage = self.stages[key]
        stage.status = "run"
        stage.total = total
        stage.current = 0
        stage.detail = detail
        stage.started_at = time.monotonic()
        self._refresh()

    def tick(self, key: str, current: int, total: int | None = None, detail: str = "") -> None:
        stage = self.stages[key]
        stage.current = current
        if total is not None:
            stage.total = total
        if detail:
            stage.detail = detail
        self._refresh()

    def finish(self, key: str, detail: str = "") -> None:
        stage = self.stages[key]
        if stage.started_at is not None:
            stage.seconds = time.monotonic() - stage.started_at
        stage.status = "done"
        stage.detail = detail
        if stage.total:
            stage.current = stage.total
        self._refresh()

    def fraction_done(self) -> float:
        if all(stage.status == "done" for stage in self.stages.values()):
            return 1.0
        done = 0.0
        for stage in self.stages.values():
            if stage.status == "done":
                done += stage.weight
            elif stage.status == "run":
                if stage.total:
                    done += stage.weight * min(1.0, stage.current / stage.total)
                else:
                    done += stage.weight * 0.15
        return min(0.99, done)

    def remaining_seconds(self) -> float | None:
        elapsed = time.monotonic() - self.started_at
        frac = self.fraction_done()
        if frac < 0.08:
            return max(0.0, self.estimated_seconds - elapsed)
        projected = elapsed / max(frac, 1e-6)
        return max(0.0, projected - elapsed)

    def _refresh(self) -> None:
        frac = self.fraction_done()
        if self._bar is not None and self._bar_task is not None:
            self._bar.update(self._bar_task, completed=frac)
        if self._live:
            self._live.update(self.render())

    def render(self) -> RenderableType:
        header = Text.assemble(
            ("videoclean", "bold cyan"),
            ("  •  ", "dim"),
            (f"job {self.job_id}", "bold"),
        )
        sub = Text(self.headline, style="dim")

        table = Table.grid(padding=(0, 2))
        table.add_column("mark", width=6)
        table.add_column("title", min_width=24)
        table.add_column("info", min_width=28)
        table.add_column("time", justify="right", width=8)
        for stage in self.stages.values():
            mark, style = _mark(stage.status)
            info = stage.detail
            if stage.status == "run" and stage.total and not stage.detail:
                info = f"{stage.current}/{stage.total}"
            elapsed = ""
            if stage.status == "done":
                elapsed = _fmt_seconds(stage.seconds)
            elif stage.status == "run" and stage.started_at:
                elapsed = _fmt_seconds(time.monotonic() - stage.started_at)
            table.add_row(
                Text(mark, style=style),
                Text(stage.title, style=style),
                Text(info, style="dim"),
                Text(elapsed, style="dim"),
            )

        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        remaining = 0.0 if self.finished_at else self.remaining_seconds()
        finish_at = datetime.now(timezone.utc).astimezone() + timedelta(seconds=remaining or 0)
        footer = Text.assemble(
            ("elapsed ", "dim"),
            (_fmt_seconds(elapsed), "bold"),
            ("   remaining ", "dim"),
            (_fmt_seconds(remaining), "bold cyan"),
            ("   finish ~ ", "dim"),
            (finish_at.strftime("%H:%M:%S") if self.finished_at is None else "done", "bold"),
        )
        body = Group(header, sub, Text(""), table, Text(""), self._bar, Text(""), footer)
        return Panel(body, border_style="cyan", padding=(1, 2))


def _mark(status: str) -> tuple[str, str]:
    return {
        "pending": ("·", "dim"),
        "run": ("▶", "yellow"),
        "done": ("✓", "green"),
        "error": ("✗", "red"),
    }.get(status, ("·", "dim"))
