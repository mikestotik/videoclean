from __future__ import annotations


class SilentProgress:
    def start(self, key: str, total: int = 0, detail: str = "") -> None:
        return None

    def tick(self, key: str, current: int, total: int | None = None, detail: str = "") -> None:
        return None

    def finish(self, key: str, detail: str = "") -> None:
        return None
