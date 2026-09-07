from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from videoclean.domain.tracks import Track

# current, total, detail — CLI heartbeat while a detector is in a long loop
OnProgress = Callable[[int, int, str], None]


class Detector(Protocol):
    """Finds overlay tracks. Does not inpaint. Does not write files."""

    name: str

    def status(self) -> str:
        """ready | unavailable: reason"""
        ...

    def discover(
        self,
        frames: list[np.ndarray],
        queries: list[str],
        on_progress: OnProgress | None = None,
    ) -> list[Track]:
        ...
