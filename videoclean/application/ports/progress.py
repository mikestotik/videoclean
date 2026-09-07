from __future__ import annotations

from typing import Protocol


class ProgressPort(Protocol):
    def start(self, key: str, total: int = 0, detail: str = "") -> None: ...

    def tick(self, key: str, current: int, total: int | None = None, detail: str = "") -> None: ...

    def finish(self, key: str, detail: str = "") -> None: ...
